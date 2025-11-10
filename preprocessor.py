import os
from typing import Optional, List, Tuple, Callable
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import time
import pickle
import tensorflow as tf

class Preprocessor:
    """Pré-processador base.

    - lag/lead como inteiros são expandidos para ranges [1..N] quando apropriado.
    - feature_cols/target_cols definem bases permitidas e servem como seleção no export.
    - Nenhuma coluna é removida dos dados; seleção ocorre apenas na exportação.
    """
    def __init__(
        self,
        lag: int,
        lead: int,
        country_list: Optional[List[str]] = None,
        *,
        model_name: str = "linear",
        data_dir: str = "data/processed",
        feature_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
    ):
        self.lag = lag
        self.lead = lead
        self.country_list = list(country_list)
        self.model_name = model_name
        self.data_dir = data_dir
        self.data_dir = self.data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.feature_cols: List[str] = list(feature_cols) if feature_cols else []
        self.target_cols: List[str] = list(target_cols) if target_cols else []

        self.norm_objects = {}
        self.encod_objects = {}
        self.df_base = pd.DataFrame()

    def _expand_steps(self, steps, default_max: Optional[int]) -> List[int]:
        """Normaliza passos: int→[1..N], None→[1..default_max], lista→como está."""
        if isinstance(steps, int):
            return list(range(1, steps + 1)) if steps > 0 else [1]
        if steps is None and isinstance(default_max, int) and default_max > 0:
            return list(range(1, default_max + 1))
        if isinstance(steps, (list, tuple)):
            return list(steps)
        return [1]

    def load_data(self, raw_dir: Optional[str] = None) -> pd.DataFrame:
        """Carrega Parquet unificado em data/raw (ou raw_dir) e atualiza self.df_base."""
        base_raw = raw_dir or os.path.join('data', 'raw')
        unified_path = os.path.join(base_raw, f'raw_dataset.parquet')
        if not os.path.exists(unified_path):
            raise FileNotFoundError(f"Arquivo unificado não encontrado: {unified_path}. Execute a coleta primeiro.")
        df = pd.read_parquet(unified_path, engine='pyarrow')
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        if self.country_list and 'country' in df.columns:
            df = df[df['country'].isin(self.country_list)].copy()
        sort_cols = [c for c in ['country', 'datetime'] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
            
        # Filtrando Colunas apenas para as necessárias
        cols = list(set([c for c in self.feature_cols + self.target_cols if c in df.columns]))
        df = df.loc[:, ~df.columns.duplicated()]  # optional: remove duplicates
        df = df[cols]

        self.df_base = df
        return self.df_base

    def encode(self, encode_cols, encode_method='label', train=True):
        """Codifica colunas de forma não destrutiva e atualiza self.df_base.
        
        - 'label': usa LabelEncoder persistente por coluna, com fallback para categorias desconhecidas.
        - 'time_cycle': adiciona features cíclicas baseadas em datetime.
        """
        if self.df_base is None or self.df_base.empty:
            print("⚠️ df_base vazio. Chame load_data() primeiro.")
            return self.df_base

        df = self.df_base.copy()

        if not hasattr(self, "encod_objects"):
            self.encod_objects = {}

        # ========= LABEL ENCODING ========= #
        if encode_method == 'label':
            if isinstance(encode_cols, str):
                encode_cols = [encode_cols]

            for col in encode_cols:
                # Reuse or create new encoder
                if not train and col in self.encod_objects:
                    le = self.encod_objects[col]['encoder']
                else:
                    le = LabelEncoder()

                # Fit only on train
                if train:
                    le.fit(df[col].astype(str).fillna("__nan__"))
                    self.encod_objects[col] = {'encoder': le}

                # Transform with fallback for unseen labels
                known = set(le.classes_)
                s = df[col].astype(str).fillna("__nan__")
                s = s.apply(lambda x: x if x in known else "__unknown__")
                if "__unknown__" not in known:
                    le.classes_ = np.append(le.classes_, "__unknown__")
                df[col] = le.transform(s)

        # ========= TIME CYCLE ENCODING ========= #
        elif encode_method == 'time_cycle':
            col = encode_cols if isinstance(encode_cols, str) else encode_cols[0]
            if col not in df.columns:
                print(f"Coluna {col} não encontrada para time_cycle.")
                self.df_base = df
                return df

            dt = pd.to_datetime(df[col], utc=True)
            current_year = time.localtime().tm_year
            for period, max_val in {
                "year": current_year, "month": 12, "day": 31, "hour": 24, "minute": 60
            }.items():
                df[f"{period}_sin"] = np.sin(2 * np.pi * getattr(dt.dt, period) / max_val)
                df[f"{period}_cos"] = np.cos(2 * np.pi * getattr(dt.dt, period) / max_val)

            self.encod_objects["time_cycle"] = {'encode_cols': col}
            self.feature_cols.extend([
                f"{p}_{a}" for p in ["year", "month", "day", "hour", "minute"] for a in ["sin", "cos"]
            ])

        else:
            raise ValueError(f"encode_method '{encode_method}' não suportado.")

        self.df_base = df
        return self.df_base


    def decode(self, encode_method: str = 'label', target_col: Optional[str] = None) -> pd.DataFrame:
        """Reverte codificações suportadas (label, time_cycle)."""
        if self.df_base is None or self.df_base.empty:
            print("df_base vazio. Nada para decodificar.")
            return self.df_base
        df = self.df_base.copy()
        if encode_method == 'label':
            info = self.encod_objects.get('label')
            if not info:
                print("Nenhuma informação de label encoding salva.")
                return self.df_base
            col = info['encode_cols']
            le: LabelEncoder = info['label_encoder']
            placeholder = info.get('na_placeholder', '__NA__')
            try:
                inv = le.inverse_transform(df[col].astype(int))
                # mapeia placeholder de volta para NaN
                inv = pd.Series(inv).replace(placeholder, np.nan).values
                df[col] = inv
            except Exception as e:
                print(f"Falha ao decodificar label para coluna {col}: {e}")
        elif encode_method == 'time_cycle':
            if 'year' not in df.columns:
                print("Componentes de tempo ausentes para reconstrução.")
                return self.df_base
            tgt = target_col or 'decoded_datetime'
            def _recover_component(sin_col, cos_col, period, offset):
                if sin_col not in df.columns or cos_col not in df.columns:
                    return pd.Series([np.nan] * len(df))
                ang = np.arctan2(df[sin_col], df[cos_col])
                ang = (ang + 2 * np.pi) % (2 * np.pi)
                idx = np.round((ang / (2 * np.pi)) * period).astype('Int64') % period
                return idx + offset
            month = _recover_component('month_sin', 'month_cos', 12, 1)
            day = _recover_component('day_sin', 'day_cos', 31, 1)
            hour = _recover_component('hour_sin', 'hour_cos', 24, 0)
            minute = _recover_component('minute_sin', 'minute_cos', 60, 0)
            year = df['year'] if 'year' in df.columns else pd.Series([np.nan] * len(df))
            dt = pd.to_datetime({
                'year': year.astype('Int64'),
                'month': month.astype('Int64'),
                'day': day.astype('Int64'),
                'hour': hour.astype('Int64'),
                'minute': minute.astype('Int64'),
            }, errors='coerce', utc=True)
            df[tgt] = dt
        else:
            print(f"encode_method '{encode_method}' não suportado para decode.")
        self.df_base = df
        return self.df_base

    def normalize(self, value_cols: List[str], normalization_method: str = 'minmax', train: bool = False) -> pd.DataFrame:
        """Normaliza colunas e atualiza self.df_base.

        Args:
            value_cols (list[str]): colunas a normalizar.
            normalization_method (str): 'minmax' ou 'standard'.
            train (bool): se True, ajusta o scaler (fit); se False, apenas aplica (transform).
        """
        if self.df_base is None or self.df_base.empty:
            print("⚠️ df_base vazio. Chame load_data() primeiro.")
            return self.df_base

        df = self.df_base.copy()

        # Escolha do scaler
        if normalization_method == 'minmax':
            scaler_class = MinMaxScaler
        elif normalization_method == 'standard':
            scaler_class = StandardScaler
        else:
            raise ValueError("normalization_method deve ser 'minmax' ou 'standard'")

        # Cria dicionário de normalizadores se não existir
        if not hasattr(self, "norm_objects"):
            self.norm_objects = {}

        # Se já existe um scaler e estamos em modo de uso (não treino)
        if not train and normalization_method in self.norm_objects:
            scaler = self.norm_objects[normalization_method]['scaler']
            df[value_cols] = scaler.transform(df[value_cols])

        # Caso contrário: cria e ajusta novo scaler
        else:
            scaler = scaler_class()
            df[value_cols] = scaler.fit_transform(df[value_cols])
            # salva para reutilização futura
            self.norm_objects[normalization_method] = {
                'value_cols': value_cols,
                'scaler': scaler
            }

        self.df_base = df
        return self.df_base
    def normalize_splits(self, value_cols: List[str], normalization_method: str = 'minmax', train: bool = False) -> dict:
        """Normaliza os conjuntos de treino, validação e teste."""
        if not self.splits:
            print("Nenhum conjunto dividido encontrado.")
            return {}
        normalized_splits = {}
        for split_name, split_df in self.splits.items():
            self.df_base = split_df
            if train and split_name == 'train':
                train = True
            else:
                train = False
            normalized_df = self.normalize(value_cols=value_cols, normalization_method=normalization_method, train=train)
            normalized_splits[split_name] = normalized_df
        self.splits = normalized_splits
        return normalized_splits

    def denormalize(self, normalization_method: str = 'minmax') -> pd.DataFrame:
        """Reverte normalização usando metadados salvos."""
        if self.df_base is None or self.df_base.empty:
            print("df_base vazio. Nada para denormalizar.")
            return self.df_base
        info = self.norm_objects.get(normalization_method)
        if not info:
            print(f"Nenhum scaler salvo para o método '{normalization_method}'.")
            return self.df_base
        cols: List[str] = info['value_cols']
        scaler = info['scaler']
        df = self.df_base.copy()
        try:
            df[cols] = scaler.inverse_transform(df[cols])
        except Exception as e:
            print(f"Falha ao denormalizar colunas {cols}: {e}")
            return self.df_base
        self.df_base = df
        return self.df_base

    def save_df_base(self, filename: Optional[str] = None, compression: Optional[str] = None, partition_by: Optional[List[str]] = None) -> Optional[str]:
        """Salva self.df_base em Parquet dentro de data_dir/{model_name}."""
        if self.df_base is None or self.df_base.empty:
            print("df_base vazio. Nada para salvar.")
            return None
        comp = compression
        if comp is None:
            comp = 'zstd'
        filename = "raw_dataset.parquet"
        out_path = os.path.join(self.data_dir, filename)
        df = self.df_base.copy()
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        try:
            if partition_by:
                df.to_parquet(out_path, engine='pyarrow', compression=comp, index=False, partition_cols=partition_by)
            else:
                df.to_parquet(out_path, engine='pyarrow', compression=comp, index=False)
            print(f"[SALVO] df_base: {len(df):,} linhas → {out_path}")
            return out_path
        except Exception as e:
            print(f"Falha ao salvar df_base em {out_path}: {e}")
            return None
    
    def split_train_val_test(self, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15, time_col: str = 'datetime') -> Optional[dict]:
        """Divide df_base em conjuntos de treino, validação e teste com base em time_col."""
        if self.df_base is None or self.df_base.empty:
            print("df_base vazio. Nada para dividir.")
            return None
        if not np.isclose(train_size + val_size + test_size, 1.0):
            print("train_size, val_size e test_size devem somar 1.0")
            return None
        df = self.df_base.copy()
        if time_col not in df.columns:
            print(f"Coluna de tempo '{time_col}' não encontrada em df_base.")
            return None
        df = df.sort_values(time_col).reset_index(drop=True)
        n = len(df)
        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)
        splits = {
            'train': df.iloc[:train_end].reset_index(drop=True),
            'val': df.iloc[train_end:val_end].reset_index(drop=True),
            'test': df.iloc[val_end:].reset_index(drop=True),
        }
        for split_name, split_df in splits.items():
            print(f"[DIVIDIDO] {split_name}: {len(split_df):,} linhas")
        self.splits = splits
        return splits
    
    def save_instance(self, path: str, name: str):
        """Save the current instance as a pickle file."""
        path = os.path.join(path, f"{name}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Instance saved at {path}")

    @classmethod
    def load_instance(cls, path: str):
        """Load a saved class instance from a pickle file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ File not found: {path}")
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"❌ Loaded object is not of type {cls.__name__}")
        print(f"✅ Instance loaded from {path}")
        return obj
    

class LinearPreprocessor(Preprocessor):
    """Pré-processador linear: gera matriz flat (lags/leads) e exporta apenas Parquet."""

    def build_flat_matrix(
        self,
        value_cols: Optional[List[str]] = None,
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
        feats = value_cols or self.feature_cols
        tgts = self.target_cols
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
        num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        drop_feats = [c for c in feats if c not in num_cols]
        drop_tgts = [c for c in tgts if c not in num_cols]
        if drop_feats:
            print(f"[INFO] Ignorando features não numéricas/bool em build_flat_matrix: {drop_feats}")
        if drop_tgts:
            print(f"[INFO] Ignorando targets não numéricos/bool em build_flat_matrix: {drop_tgts}")
        feats = [c for c in feats if c in num_cols]
        tgts = [c for c in tgts if c in num_cols]
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
                cname = f"{col}_lag{k}"
                df[cname] = df.groupby("_group_id", group_keys=False, sort=False)[col].shift(k)
                new_cols.append(cname)
            # Lags mascarados (padding)
            for k in padded_lags:
                cname = f"{col}_lag{k}"
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

        # ---- Drop NA ----
        df.dropna(subset=new_cols, inplace=True) if dropna else None

        df.drop(columns=["_group_id"], inplace=True, errors="ignore")

        # Atualiza atributos
        self.df_base = df
        self.feature_cols.extend([c for c in new_cols if "_lag" in c and c not in self.feature_cols])
        self.target_cols.extend([c for c in new_cols if "_lead" in c and c not in self.target_cols])

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
            fraw = [c for c in self.feature_cols if c in num_cols]
            traw = [c for c in self.target_cols if c in num_cols]
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
    


import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from lightning.pytorch import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from typing import List, Optional


class TFTPreprocessor(Preprocessor):
    """
    Preprocessador específico para o modelo Temporal Fusion Transformer (PyTorch Forecasting).
    Herdando de Preprocessor, apenas adiciona a etapa final de estruturação e salvamento
    dos splits no formato compatível com o PyTorch Forecasting.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str,
        feature_cols: List[str],
        target_cols: List[str],
        country_list: List[str],
        seq_len: int,
        lead: int,
    ):
        # Corrigido: alinhar com assinatura de Preprocessor
        super().__init__(
            lag=seq_len,
            lead=lead,
            country_list=country_list,
            model_name=model_name,
            data_dir=data_dir,
            feature_cols=feature_cols,
            target_cols=target_cols,
        )
        self.seq_len = seq_len
        self.lead = lead


    def build_tft_parquets(
        self,
        group_cols: Optional[List[str]] = ["country"],
        time_col: str = "datetime",
        dropna: bool = True,
    ):
        """
        Estrutura os splits existentes (já criados na classe-base) para uso no TFT e salva em parquet.
        Simples e direto:
        - Ordena por (group_cols + time_col)
        - Opcionalmente remove nulos nas colunas críticas [time_col] + group_cols + target_cols
        - Define '_group_id' e calcula 'time_idx' por grupo via cumcount() (0..N-1 por série)
        - Salva parquet por split + meta.json (x_dim, y_dim, seq_len, lead, feature_cols, target_cols, path)
        """
        if not hasattr(self, "splits") or not self.splits:
            raise ValueError("Os splits ainda não foram gerados. Execute split_train_val_test() primeiro.")
        import json
        out = {}
        for name, df in self.splits.items():
            df = df.copy()
            # tipos e ordenação
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            sort_cols = (group_cols or []) + [time_col]
            df = df.sort_values(sort_cols).reset_index(drop=True)

            # drop nulos básico
            if dropna:
                subset_cols = ([time_col] if time_col else []) + (group_cols or []) + (self.target_cols or [])
                present = [c for c in subset_cols if c in df.columns]
                before = len(df)
                df = df.dropna(subset=present).reset_index(drop=True)
                if before - len(df) > 0:
                    print(f"🧹 Drop NA ({name}): {before - len(df)} linhas removidas nas colunas {present}.")

            # id de grupo e time_idx por grupo
            if group_cols:
                df["_group_id"] = df[group_cols].astype(str).agg("_".join, axis=1)
            else:
                df["_group_id"] = "global"

            # contador sequencial por grupo (não global)
            df["time_idx"] = df.groupby("_group_id").cumcount().astype("int64")

            # manter apenas colunas numéricas/bool para valores; preservar colunas essenciais
            num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
            essential = set((group_cols or []) + [time_col, "_group_id", "time_idx"] + (self.target_cols or []))
            cols_keep = [c for c in df.columns if (c in num_cols) or (c in essential)]
            if len(cols_keep) != len(df.columns):
                removed = [c for c in df.columns if c not in cols_keep]
                if removed:
                    print(f"[INFO] TFT: removendo colunas não numéricas/bool do split '{name}': {removed}")
            df = df[cols_keep]

            # salvar parquet
            path = os.path.join(self.data_dir, f"tft_dataset_{name}.parquet")
            df.to_parquet(path, index=False)
            max_local_time_idx = df.groupby('_group_id')['time_idx'].max().max()
            print(f"💾 Split '{name}' salvo em {path} ({df.shape[0]} linhas, grupos={df['_group_id'].nunique()}, max local time_idx={max_local_time_idx}).")

            # meta json
            meta = {
                "seq_len": int(getattr(self, 'seq_len', self.lag)),
                "lead": int(getattr(self, 'lead', 0)),
                "x_dim": len(self.feature_cols),
                "y_dim": len(self.target_cols),
                "feature_cols": list(self.feature_cols),
                "target_cols": list(self.target_cols),
                "parquet_path": path,
                "basename": f"tft_dataset_{name}",
                "groups": int(df['_group_id'].nunique()),
                "max_local_time_idx": int(max_local_time_idx),
            }
            meta_path = os.path.join(self.data_dir, f"tft_dataset_{name}.meta.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            out[name] = {"path": path, "meta": meta}
        return out


    def load_tft_dataset(
        self,
        split_name: str,
        target_col: str,
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
            target=target_col,
            group_ids=["_group_id"],
            max_encoder_length=self.seq_len,
            max_prediction_length=self.lead,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=self.target_cols,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        print(f"📦 TimeSeriesDataSet ({split_name}) criado com {len(df)} amostras.")
        return ds