
from preprocessor import Preprocessor
import os
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
        - Orders by group columns + datetime
        - Creates '_group_id'
        - Creates 'time_idx' = cumcount() per group
        - Saves parquets for each split
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

            print(f"✔️ Saved {name}: {path} (rows={len(df)}, groups={df['_group_id'].nunique()})")

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

        ds =  TimeSeriesDataSet(
            data = df,
            time_idx="time_idx",
            target=self.target_cols[0],
            group_ids=self.group_cols,
            min_encoder_length=1,
            max_encoder_length=self.lag,
            min_prediction_length=1,
            max_prediction_length=self.lead,
            static_categoricals=self.group_cols,
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx", "year", "month", "day", "hour", "minute"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=self.target_cols,
            target_normalizer=GroupNormalizer(
                groups=self.group_cols, transformation="softplus"
            ), 
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        print(f"📦 TimeSeriesDataSet ({split_name}) criado com {len(df)} amostras.")
        return ds
    