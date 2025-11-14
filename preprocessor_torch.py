
from preprocessor import Preprocessor
import os
import json
import datetime as dt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from lightning.pytorch import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting.data import GroupNormalizer
from typing import List, Optional


class TFTPreprocessor(Preprocessor):
    """
    Preprocessador específico para o modelo Temporal Fusion Transformer (PyTorch Forecasting).
    Herdando de Preprocessor, apenas adiciona a etapa final de estruturação e salvamento
    dos splits no formato compatível com o PyTorch Forecasting.
    """

    def __init__(self, group_cols=["country"], time_col="datetime", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Instanciando parâmetros de colunas
        self.group_cols = group_cols
        self.time_col = time_col
        
    def build_tft_parquets(
        self,
    ):
        """
        Gera parquets para cada split no formato esperado pelo TimeSeriesDataSet e cria
        um meta.json semelhante aos preprocessadores Keras (linear/lstm), contendo:
        - lag / lead
        - x_dim / y_dim
        - feature_cols / target_cols
        - group_cols / time_col
        - rows / groups
        - min_time / max_time
        - parquet_path / basename
        - time_part_cols (se existirem colunas de partes de tempo)

        Passos:
        1. Ordena por group_cols + datetime
        2. Cria '_group_id'
        3. Cria 'time_idx' cumulativo por grupo
        4. Salva parquet
        5. Salva meta JSON adjacente
        """

        if not hasattr(self, "splits") or not self.splits:
            raise ValueError("Splits not found. Run split_train_val_test() first.")

        out = {}
        for name, df in self.splits.items():
            df = df.copy()

            # Force datetime
            df[self.time_col] = pd.to_datetime(df[self.time_col], utc=True)

            # Sort
            sort_cols = (self.group_cols or []) + [self.time_col]
            df = df.sort_values(sort_cols).reset_index(drop=True)

            # Build group id
            if self.group_cols:
                df["_group_id"] = df[self.group_cols].astype(str).agg("_".join, axis=1)
            else:
                df["_group_id"] = "global"

            # Build time index
            df["time_idx"] = df.groupby("_group_id").cumcount().astype("int32")

            # Save parquet
            path = os.path.join(self.data_dir, f"tft_dataset_{name}.parquet")
            df.to_parquet(path, index=False)

            groups_n = df["_group_id"].nunique()
            rows_n = len(df)
            min_time = df[self.time_col].min()
            max_time = df[self.time_col].max()

            # Infer time part columns (common naming from encode('time_parts'))
            time_part_cols = [c for c in df.columns if c in {"year", "month", "day", "hour", "minute"}]

            # Infer numeric/continuous columns
            try:
                num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            except Exception:
                num_cols = []

            # Build TFT schema fields
            static_categoricals = list(self.group_cols or [])
            time_varying_known_categoricals = []  # adjust if you later add any KV categoricals
            # known reals: time_idx + time parts + any other numeric feature (excluding targets/groups)
            extra_known_reals = [
                c for c in (self.feature_cols or [])
                if c not in (self.target_cols or [])
                and c not in (self.group_cols or [])
                and c in df.columns
                and str(df[c].dtype).startswith(("int", "float"))
            ]
            time_varying_known_reals = ["time_idx"] + time_part_cols + [c for c in extra_known_reals if c not in time_part_cols]
            time_varying_unknown_categoricals = []
            time_varying_unknown_reals = list(self.target_cols or [])

            dtypes_map = {c: str(df[c].dtype) for c in df.columns}

            meta = {
                "lag": int(getattr(self, "lag", 0)),
                "lead": int(getattr(self, "lead", 0)),
                "x_dim": int(len(self.feature_cols or [])),
                "y_dim": int(len(self.target_cols or [])),
                "feature_cols": list(self.feature_cols or []),
                "target_cols": list(self.target_cols or []),
                "group_cols": list(self.group_cols or []),
                "time_col": self.time_col,
                "rows": rows_n,
                "groups": groups_n,
                "min_time": None if min_time is pd.NaT else min_time.isoformat(),
                "max_time": None if max_time is pd.NaT else max_time.isoformat(),
                "time_part_cols": time_part_cols,
                "continuous_cols": [c for c in num_cols if c != "time_idx"],
                "dtypes": dtypes_map,
                "parquet_path": path.replace(os.sep, "/"),
                "basename": os.path.splitext(os.path.basename(path))[0],
                "generated_at": dt.datetime.utcnow().isoformat() + "Z",
                "backend": "pytorch_forecasting",
                "kind": "tft_dataset_split",
                "tft_schema": {
                    "static_categoricals": static_categoricals,
                    "time_varying_known_categoricals": time_varying_known_categoricals,
                    "time_varying_known_reals": time_varying_known_reals,
                    "time_varying_unknown_categoricals": time_varying_unknown_categoricals,
                    "time_varying_unknown_reals": time_varying_unknown_reals,
                    "target": (self.target_cols or [None])[0],
                    "group_ids": list(self.group_cols or []),
                    "max_encoder_length": int(getattr(self, "lag", 0)),
                    "max_prediction_length": int(getattr(self, "lead", 0)),
                },
            }

            meta_path = os.path.join(self.data_dir, f"tft_dataset_{name}.meta.json")
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] Falha ao salvar meta JSON ({name}): {e}")

            print(
                f"✔️ Saved {name}: {path} (rows={rows_n}, groups={groups_n}) | meta: {meta_path}"
            )

            out[name] = path

        return out

    def load_tft_dataset(
        self,
        split_name: str,
        known_reals: Optional[List[str]] = None,
        return_df: bool = False,
    ):
        """
        Carrega o parquet salvo como DataFrame ou cria um TimeSeriesDataSet compatível com o TFT PyTorch.

        Args:
            split_name: 'train' | 'val' | 'test' (parte do nome do arquivo parquet gerado)
            target_col: coluna alvo principal (string)
            known_reals: lista de features conhecidas no tempo (overrides self.feature_cols quando fornecida)
            return_df: se True retorna o DataFrame bruto em vez do TimeSeriesDataSet

        Retorna:
            DataFrame (quando return_df=True) ou TimeSeriesDataSet
        """
        path = os.path.join(self.data_dir, f"tft_dataset_{split_name}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        df = pd.read_parquet(path)

        if return_df:
            print(f"📥 Parquet '{split_name}' carregado ({len(df)} linhas) — retornando DataFrame.")
            return df

        # determina known/unknown reals
        known_reals = known_reals or [c for c in (self.feature_cols or []) if c not in (self.target_cols or [])]

        ds = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=self.target_cols[0],
            group_ids=self.group_cols,

            min_encoder_length=self.lag,
            max_encoder_length=self.lag,
            min_prediction_length=self.lead,
            max_prediction_length=self.lead,

            static_categoricals=self.group_cols,

            time_varying_known_reals=[
                "time_idx", "year", "month", "day", "hour", "minute"
            ],

            time_varying_unknown_reals=self.target_cols,

            # NORMALIZA APENAS O TARGET
            target_normalizer=GroupNormalizer(
                groups=self.group_cols,
            ),

            # IMPORTANTÍSSIMO PARA NÃO FERRAR SUAS FEATURES
            add_target_scales=False,

            # Mantenha para melhora no treinamento
            add_relative_time_idx=True,
            add_encoder_length=True,
        )

        print(f"📦 TimeSeriesDataSet ({split_name}) criado com {len(df)} amostras.")
        return ds
    