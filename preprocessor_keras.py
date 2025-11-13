import os
from typing import Optional, List, Tuple, Callable
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import time
import pickle
import tensorflow as tf
from preprocessor import Preprocessor   

class LinearPreprocessor(Preprocessor):
    """Pré-processador linear: gera matriz flat (lags/leads) e exporta apenas Parquet."""

    def build_flat_matrix(
        self,
        lags: Optional[int] = None,
        leads: Optional[int] = None,
        seq_len: Optional[int] = None,
        mask_value: float = 0.0,
        dropna: bool = True,
        group_cols: Optional[List[str]] = None,
        time_col: str = "datetime",
    ) -> pd.DataFrame:
        import pandas as pd
        import numpy as np

        if self.df_base is None or self.df_base.empty:
            print("df_base vazio. Chame load_data() primeiro.")
            return self.df_base

        df = self.df_base.copy()
        # features e targes numéricas -> self.num_cols
        feats = [c for c in self.feature_cols if c in self.num_cols]
        tgts = [c for c in self.target_cols if c in self.num_cols]
        if not feats:
            raise ValueError("Nenhuma coluna de feature informada.")
        if not tgts:
            raise ValueError("Nenhum target informado.")

        group_cols = group_cols or [c for c in ["country"] if c in df.columns]
        if time_col not in df.columns:
            raise ValueError(f"Coluna temporal '{time_col}' não encontrada no DataFrame.")

        # Ordena
        sort_cols = (group_cols or []) + [time_col]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        if group_cols:
            df["_group_id"] = df[group_cols].astype(str).agg("_".join, axis=1)
        else:
            df["_group_id"] = "global"

        # Apenas colunas numéricas/bool para a construção (não altera self.feature_cols/self.target_cols)
        numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        drop_feats = [c for c in feats if c not in numeric_cols]
        drop_tgts = [c for c in tgts if c not in numeric_cols]
        if drop_feats:
            print(f"[INFO] Ignorando features não numéricas/bool em build_flat_matrix: {drop_feats}")
        if drop_tgts:
            print(f"[INFO] Ignorando targets não numéricos/bool em build_flat_matrix: {drop_tgts}")
        feats = [c for c in feats if c in numeric_cols]
        tgts = [c for c in tgts if c in numeric_cols]
        if not feats:
            raise ValueError("Nenhuma feature numérica/bool disponível para build_flat_matrix.")
        if not tgts:
            raise ValueError("Nenhum target numérico/bool disponível para build_flat_matrix.")

        lag_steps = list(range(1, (lags or self.lag or 0) + 1))
        lead_steps = list(range(1, (leads or self.lead or 0) + 1))
        new_cols = []

        # Determina redução
        if seq_len is not None and seq_len < len(lag_steps):
            active_lags = lag_steps[:seq_len]
            padded_lags = lag_steps[seq_len:]
            print(f"[ℹ️] Reduzindo lags: usando {len(active_lags)} e mascarando {len(padded_lags)} restantes com {mask_value}")
        else:
            active_lags = lag_steps
            padded_lags = []

        # ---- Lags ----
        for col in feats:
            if col not in df.columns:
                print(f"[WARN] Coluna de feature '{col}' não encontrada.")
                continue
            # Lags ativos
            for k in active_lags:
                # remove _noise from column name for lag naming
                col_name = col.replace("_noise", "")
                cname = f"{col_name}_lag{k}"
                df[cname] = df.groupby("_group_id", group_keys=False, sort=False)[col].shift(k)
                new_cols.append(cname)
            # Lags mascarados (padding)
            for k in padded_lags:
                cname = f"{col_name}_lag{k}"
                df[cname] = mask_value
                new_cols.append(cname)

        # ---- Leads ----
        for tgt in tgts:
            if tgt in df.columns:
                for k in lead_steps:
                    cname = f"{tgt}_lead{k}"
                    df[cname] = df.groupby("_group_id", group_keys=False, sort=False)[tgt].shift(-k)
                    new_cols.append(cname)
            else:
                print(f"[WARN] Target '{tgt}' não encontrado. Ignorando leads.")

        # Remove colunas noise adicionadas anteriormente
        noise_remove_cols = [c for c in self.num_cols if "_noise" in c]
        self.num_cols = [c for c in self.num_cols if c not in noise_remove_cols]
        
        # Renomeando coluna sem noise pela com noise
        for col in feats:
            tmp = col.endswith("_noise")
            if tmp:
                base_col = col.replace("_noise", "")
                if base_col in df.columns:
                    self.feature_cols.append(base_col)
                    self.feature_cols.remove(col)
                    df[base_col] = df[col]
                    df.drop(columns=[col], inplace=True)

        # ---- Drop NA ----
        df.dropna(subset=new_cols, inplace=True) if dropna else None

        df.drop(columns=["_group_id"], inplace=True, errors="ignore")

        # Atualiza atributos
        self.df_base = df
        self.target_cols_real = []
        self.feature_cols_real = self.feature_cols.copy()
        self.feature_cols_real.extend(self.num_cols)  # mantém colunas numéricas originais
        self.feature_cols_real.extend([c for c in new_cols if "_lag" in c and c not in self.feature_cols])
        self.target_cols_real.extend([c for c in new_cols if "_lead" in c and c not in self.target_cols])

        return self.df_base

    def build_flat_matrices_splits(self, *args, **kwargs) -> Optional[dict]:
        """Constrói matrizes flat para cada split (train/val/test)."""
        if not self.splits:
            print("Nenhum conjunto dividido encontrado.")
            return None
        built_splits = {}
        for split_name, split_df in self.splits.items():
            self.df_base = split_df
            built_df = self.build_flat_matrix(*args, **kwargs)
            built_splits[split_name] = built_df
        self.splits = built_splits
        return built_splits
    
    def save_linear_splits_parquet(self, basename: str = "linear_dataset", row_group_size: int = 200_000, verbose: bool = True) -> dict:
        """
        Salva cada split (train/val/test) como Parquet + meta.json com x_dim/y_dim/colunas.
        De-duplica listas de colunas e evita sobreposição entre features e targets.
        Usa pyarrow ParquetWriter com row groups (chunks) para reduzir picos de memória.
        """
        import json, os
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        def _uniq(seq):
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            return out

        out = {}
        for split_name, df in (getattr(self, 'splits', {}) or {}).items():
            path = os.path.join(self.data_dir, f"{basename}_{split_name}.parquet")
            num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
            fraw = [c for c in self.feature_cols_real if c in num_cols]
            traw = [c for c in self.target_cols_real if c in num_cols]
            tcols = _uniq(traw)
            fcols = [c for c in _uniq(fraw) if c not in set(tcols)]
            combined_cols = fcols + tcols

            if verbose:
                print(f"[Linear:{split_name}] linhas={len(df):,}  X={len(fcols)}  Y={len(tcols)}  escrevendo em chunks de {row_group_size:,}…")

            writer = None
            try:
                n = len(df)
                for start in range(0, n, row_group_size):
                    end = min(start + row_group_size, n)
                    chunk = df.iloc[start:end][combined_cols]
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(path, table.schema, compression="snappy")
                    writer.write_table(table)
                if writer is not None:
                    writer.close()
            finally:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass

            meta = {
                "x_dim": int(len(fcols)),
                "y_dim": int(len(tcols)),
                "feature_cols": fcols,
                "target_cols": tcols,
                "parquet_path": path,
                "basename": f"{basename}_{split_name}"
            }
            with open(os.path.join(self.data_dir, f"{basename}_{split_name}.meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            out[split_name] = {"path": path, "meta": meta}
            if verbose:
                print(f"[💾] Parquet salvo: {path}")
        return out

    @staticmethod
    def load_linear_parquet_dataset(data_dir: str, split: str, batch_size: int = 256, shuffle: bool = True) -> Tuple[tf.data.Dataset, dict]:
        """
        Carrega Parquet para NumPy e monta tf.data a partir de memória (rápido).
        """

        meta_path = os.path.join(data_dir, f"linear_dataset_{split}.meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta JSON não encontrado: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        df = pd.read_parquet(meta["parquet_path"])
        X = df[meta["feature_cols"]].to_numpy("float32", copy=False)
        Y = df[meta["target_cols"]].to_numpy("float32", copy=False)
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        if shuffle:
            ds = ds.shuffle(min(len(df), 10000), seed=42, reshuffle_each_iteration=False)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, meta

import os, json, re
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional, Dict, Any


class LSTMPreprocessor(Preprocessor):
    """Pré-processador sequencial para LSTM: gera janelas 3D (N, seq_len, features) e Y (N, lead, targets)."""

    # =====================================================
    # BUILD MATRIX
    # =====================================================
    def build_sequence_matrix(
        self,
        group_cols: Optional[List[str]] = None,
        time_col: str = "datetime",
    ):
        """
        Constrói tensores X (entradas) e Y (alvos) para modelo LSTM multivariado.
        Agora suporta múltiplos passos de previsão (lead > 1).
        Somente colunas numéricas/bool são utilizadas para montar os tensores; colunas não numéricas
        em self.feature_cols/self.target_cols são ignoradas apenas nesta construção (listas originais não são alteradas).
        """
        if self.df_base is None or self.df_base.empty:
            raise ValueError("df_base vazio. Chame load_data() primeiro.")

        df = self.df_base.copy()
        feats = self.feature_cols
        tgts = self.target_cols
        if not feats:
            raise ValueError("Nenhuma coluna de feature informada.")
        if not tgts:
            raise ValueError("Nenhum target informado.")
        # Filtra somente numéricas/bool sem modificar atributos globais
        num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        drop_feats = [c for c in feats if c not in num_cols]
        drop_tgts = [c for c in tgts if c not in num_cols]
        if drop_feats:
            print(f"[INFO] Ignorando features não numéricas/bool em build_sequence_matrix: {drop_feats}")
        if drop_tgts:
            print(f"[INFO] Ignorando targets não numéricos/bool em build_sequence_matrix: {drop_tgts}")
        feats = [c for c in feats if c in num_cols]
        tgts = [c for c in tgts if c in num_cols]
        if not feats:
            raise ValueError("Nenhuma feature numérica/bool disponível para build_sequence_matrix.")
        if not tgts:
            raise ValueError("Nenhum target numérico/bool disponível para build_sequence_matrix.")
        if time_col not in df.columns:
            raise ValueError(f"Coluna temporal '{time_col}' não encontrada.")

        group_cols = group_cols or [c for c in ["country"] if c in df.columns]
        sort_cols = (group_cols or []) + [time_col]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        if group_cols:
            df["_group_id"] = df[group_cols].astype(str).agg("_".join, axis=1)
        else:
            df["_group_id"] = "global"

        seq_len = getattr(self, "lag", 24)
        lead = getattr(self, "lead", 1)

        X_list, Y_list = [], []
        for gid, g in df.groupby("_group_id", sort=False):
            g = g.reset_index(drop=True)
            if len(g) < seq_len + lead:
                continue

            X_src = g[feats].to_numpy(np.float32)
            Y_src = g[tgts].to_numpy(np.float32)

            # cria janelas deslizantes
            for i in range(len(g) - seq_len - lead + 1):
                x_win = X_src[i : i + seq_len]
                y_seq = Y_src[i + seq_len : i + seq_len + lead]  # <--- multi-step
                X_list.append(x_win)
                Y_list.append(y_seq)

        if not X_list:
            print("[WARN] Nenhuma janela gerada.")
            return {}

        X = np.stack(X_list)
        Y = np.stack(Y_list)
        print(f"[JANELAS] X={X.shape}, Y={Y.shape}, seq_len={seq_len}, lead={lead}")

        # === NEW: Capture country IDs ===
        country_ids = []
        if "country" in df.columns:
            for _, g in df.groupby("_group_id", sort=False):
                if len(g) < seq_len + lead:
                    continue
                cid = g["country"].iloc[0]
                n_win = len(g) - seq_len - lead + 1
                country_ids.extend([cid] * n_win)
        else:
            country_ids = [0] * len(X)

        country_ids = np.array(country_ids, dtype=np.int32)

        # === Store full window representation ===
        self._seq_data = pd.DataFrame({
            "X": [x.flatten() for x in X],
            "Y": [y.flatten() for y in Y],
            "country_id": country_ids,
        })
        self.feature_cols = feats
        self.target_cols = tgts
        return self._seq_data

    def build_sequence_matrix_splits(self, *args, **kwargs) -> Optional[dict]:
        """Constrói matrizes flat para cada split (train/val/test)."""
        if not self.splits:
            print("Nenhum conjunto dividido encontrado.")
            return None
        built_splits = {}
        for split_name, split_df in self.splits.items():
            self.df_base = split_df
            built_df = self.build_sequence_matrix(*args, **kwargs)
            built_splits[split_name] = built_df
        self.splits = built_splits
        return built_splits
 

    # =====================================================
    # SAVE PARQUET (streaming por splits)
    # =====================================================
    def save_splits_parquet(
        self,
        basename: str = "lstm_dataset",
        verbose: bool = True,
    ) -> dict:
        """
        Saves each DataFrame in `self.splits` (with columns ['X','Y'])
        as a Parquet file + JSON metadata.
        """

        import pyarrow.parquet as pq
        import json, os

        if not getattr(self, "splits", None):
            raise ValueError("Nenhum split encontrado. Execute split_train_val_test() antes.")

        seq_len = int(getattr(self, "seq_len", getattr(self, "lag", None) or 0))
        lead = int(getattr(self, "lead", getattr(self, "horizon", None) or 0))
        if seq_len <= 0 or lead <= 0:
            raise AttributeError("Atributos seq_len e lead/horizon devem estar definidos (>0).")

        os.makedirs(self.data_dir, exist_ok=True)
        out = {}

        for split_name, df in self.splits.items():
            required = {"X", "Y", "country_id"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Faltando colunas {missing} no DataFrame '{split_name}'.")

            pq_path = os.path.join(self.data_dir, f"{basename}_{split_name}.parquet")

            # --- Save as Parquet directly ---
            df.to_parquet(pq_path, index=False, compression="snappy")
            if verbose:
                print(f"[💾] Parquet salvo: {pq_path} ({len(df):,} linhas)")

            # --- Meta info ---
            meta = {
                "seq_len": seq_len,
                "lead": lead,
                "x_dim": len(self.feature_cols),
                "y_dim": len(self.target_cols),
                "feature_cols": getattr(self, "feature_cols", []),
                "target_cols": getattr(self, "target_cols", []),
                "parquet_path": pq_path,
                "basename": f"{basename}_{split_name}",
            }

            meta_path = os.path.join(self.data_dir, f"{basename}_{split_name}.meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            out[split_name] = {"path": pq_path, "meta": meta}

        return out

    @staticmethod
    def load_lstm_parquet_dataset(data_dir: str, split: str, batch_size: int = 64, shuffle: bool = True) -> Tuple[tf.data.Dataset, dict]:
        """
        Lê Parquet com colunas X(list<float>), Y(list<float>) e reconstrói tensores:
        X -> (seq_len, x_dim), Y -> (lead, y_dim)
        Retorna tf.data.Dataset[(X, Y)] e meta.
        """
        meta_path = os.path.join(data_dir, f"lstm_dataset_{split}.meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta JSON não encontrado: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        pq = meta["parquet_path"]
        df = pd.read_parquet(pq)

        seq_len = int(meta["seq_len"]) ; lead = int(meta["lead"]) ; x_dim = int(meta["x_dim"]) ; y_dim = int(meta["y_dim"]) 
        # Converte listas -> numpy e reshape por janela
        X_list = df["X"].to_list()
        Y_list = df["Y"].to_list()
        country_ids = df["country_id"].to_numpy(dtype=np.int32).reshape(-1, 1)

        X = np.asarray(X_list, dtype=np.float32).reshape((-1, seq_len, x_dim))
        Y = np.asarray(Y_list, dtype=np.float32).reshape((-1, lead, y_dim))

        ds = tf.data.Dataset.from_tensor_slices((
            {"num_feats": X, "country_id": country_ids},
            Y
        ))


        if shuffle:
            ds = ds.shuffle(min(len(df), 10000), seed=42, reshuffle_each_iteration=False)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, meta
    

