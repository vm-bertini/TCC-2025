import os
from typing import Optional, List, Tuple, Callable
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import time
import pickle

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
        num_cols: Optional[List[str]] = None,
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
        # Colunas numéricas/bool configuráveis (podem ser definidas externamente).
        # Se não fornecidas, serão detectadas dinamicamente na primeira normalização.
        self.num_cols: Optional[List[str]] = list(num_cols) if num_cols else None

    def _expand_steps(self, steps, default_max: Optional[int]) -> List[int]:
        """Normaliza passos: int→[1..N], None→[1..default_max], lista→como está."""
        if isinstance(steps, int):
            return list(range(1, steps + 1)) if steps > 0 else [1]
        if steps is None and isinstance(default_max, int) and default_max > 0:
            return list(range(1, default_max + 1))
        if isinstance(steps, (list, tuple)):
            return list(steps)
        return [1]

    def load_data(self, raw_dir: Optional[str] = None, size: float = 1.0, noise: bool = False) -> pd.DataFrame:
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
        if 0.0 < size < 1.0:
            partition_col = 'country' if 'country' in df.columns else None
            if partition_col:
                df = (
                    df.groupby(partition_col, group_keys=False)
                    .apply(lambda g: g.sample(frac=size, random_state=42))
                    .reset_index(drop=True)
                )
                print(f"[INFO] Carregado {len(df):,} linhas ({size*100:.1f}%) "
                    f"de {unified_path} [balanceado por '{partition_col}']")
            else:
                n_rows = int(len(df) * size)
                df = df.iloc[:n_rows].copy()
                print(f"[INFO] Carregado {n_rows:,} linhas ({size*100:.1f}%) de {unified_path}")
            
        # Filtrando Colunas apenas para as necessárias
        cols = list(set([c for c in self.feature_cols + self.target_cols if c in df.columns]))
        df = df.loc[:, ~df.columns.duplicated()]  # optional: remove duplicates
        df = df[cols]

        if noise:
            numeric_cols = self.num_cols.copy()
            noise_level = 0.01  # 1% noise
            for col in numeric_cols:
                if col in df.columns:
                    # Adiciona ruído gaussiano
                    col_std = df[col].std()
                    noise_values = np.random.normal(0, noise_level * col_std, size=len(df))
                    df[f"{col}_noise"] = df[col] + noise_values
                    # Atualiza feature_cols e num_cols
                    self.feature_cols.remove(col)
                    self.feature_cols.append(f"{col}_noise")
                    self.num_cols.append(f"{col}_noise")

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
            self.encode_time_rule = {
                "month": 12, "day": 31, "hour": 24, "minute": 60
            }
            for period, max_val in self.encode_time_rule.items():
                df[f"{period}_sin"] = np.sin(2 * np.pi * (getattr(dt.dt, period) - 1) / max_val)
                df[f"{period}_cos"] = np.cos(2 * np.pi * (getattr(dt.dt, period) - 1) / max_val)


            self.encod_objects["time_cycle"] = {'encode_rule': self.encode_time_rule, 'encode_cols': col}
            self.feature_cols.extend([
                f"{p}_{a}" for p in ["month", "day", "hour", "minute"] for a in ["sin", "cos"]
            ])
        
            df['year'] = dt.dt.year.astype('int32')
            self.feature_cols.append('year')
            self.df_base = df
            # Label encode year
            self.encode(encode_method='label', encode_cols=['year'], train=train)
            df = self.df_base

        # ========= DATETIME PARTS (DISCRETE) ========= #
        elif encode_method == 'time_parts':
            # Similar ao 'time_cycle', porém apenas adiciona colunas discretas (year, month, day, hour, minute)
            # sem transformação senoidal/cosseno.
            col = encode_cols if isinstance(encode_cols, str) else encode_cols[0]
            if col not in df.columns:
                print(f"Coluna {col} não encontrada para time_parts.")
                self.df_base = df
                return df

            dt = pd.to_datetime(df[col], utc=True)
            parts = {
                'year': dt.dt.year.astype('int32'),
                'month': dt.dt.month.astype('int16'),
                'day': dt.dt.day.astype('int16'),
                'hour': dt.dt.hour.astype('int16'),
                'minute': dt.dt.minute.astype('int16'),
            }
            for part_name, series in parts.items():
                new_col = part_name
                if new_col in df.columns:
                    # Evitar sobrescrever se já existir (ex.: múltiplas chamadas)
                    continue
                df[new_col] = series

            # Atualiza metadados
            self.encod_objects.setdefault('time_parts', {})['encode_cols'] = col

            # Garante que estas novas colunas entrem em feature_cols (evitando duplicatas)
            for c in ['year', 'month', 'day', 'hour', 'minute']:
                if c not in self.feature_cols:
                    self.feature_cols.append(c)
            # Se self.num_cols já foi explicitada, opcionalmente adiciona as partes como numéricas
            if isinstance(self.num_cols, list):
                for c in ['year', 'month', 'day', 'hour', 'minute']:
                    if c not in self.num_cols:
                        self.num_cols.append(c)

        else:
            raise ValueError(f"encode_method '{encode_method}' não suportado.")

        self.df_base = df
        return self.df_base


    def decode(self, encode_method: str, df: pd.DataFrame | None = None, target_col: str | None = None,):
        if df is None:
            df = self.df_base.copy()

        # ==========================================================
        # LABEL DECODING
        # ==========================================================
        if encode_method == "label":
            # exemplo de label decoding genérico
            obj = self.encod_objects.get(target_col, {})
            col = target_col

            if col is None or col not in df.columns:
                print("Coluna para label decode não encontrada.")
                return df

            mapper = obj.get("encoder", {})
            if mapper is None:
                print(f"Mapa inverso de label para '{col}' não encontrado.")
                return df

            df[col] = mapper.inverse_transform(df[col].astype(int))
            return df

        # ==========================================================
        # TIME CYCLE DECODING
        # ==========================================================
        elif encode_method == "time_cycle":
            # 1) Primeiro: decodifica o YEAR via LABEL (recursivo)
            #    porque agora o year foi label-encoded no encode.
            if "year" in df.columns:
                df = self.decode(
                    encode_method="label",
                    df=df,
                    target_col="year",
                )

            # se mesmo assim não tiver year, aborta
            if "year" not in df.columns:
                print("Componentes de tempo ausentes para reconstrução.")
                return df

            tgt = target_col or "datetime"

            def _recover_component(name: str, period: int, offset: int):
                sin_col = f"{name}_sin"
                cos_col = f"{name}_cos"

                if sin_col not in df.columns or cos_col not in df.columns:
                    # coluna ausente -> tudo NaN / NA
                    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

                ang = np.arctan2(df[sin_col], df[cos_col])
                ang = (ang + 2*np.pi) % (2*np.pi)

                # convert to 0-based bin index
                idx = np.floor((ang / (2*np.pi)) * period + 1e-9).astype("Int64")
                idx = idx % period

                # convert to 1-based datetime labels
                return idx + 1

            # 2) Reconstrói componentes a partir de _sin/_cos
            month = _recover_component("month", 12, 1)
            day = _recover_component("day", 31, 1)
            hour = _recover_component("hour", 24, 0)
            minute = _recover_component("minute", 60, 0)

            # year agora já está decodificado (via label) acima
            year = df["year"].astype("Int64")

            dt = pd.to_datetime(
                {
                    "year": year,
                    "month": month.astype("Int64"),
                    "day": day.astype("Int64"),
                    "hour": hour.astype("Int64"),
                    "minute": minute.astype("Int64"),
                },
                errors="coerce",
                utc=True,
            )
            df[tgt] = dt

            # se quiser limpar os *_sin / *_cos pra paridade:
            # time_cols_to_drop = [
            #     c for c in df.columns if c.endswith("_sin") or c.endswith("_cos")
            # ]
            # df.drop(columns=time_cols_to_drop, inplace=True, errors="ignore")

            return df

        else:
            print(f"encode_method '{encode_method}' não suportado para decode.")
            return df
    def normalize(self, normalization_method: str = 'minmax', train: bool = False) -> pd.DataFrame:
        """Normaliza colunas numéricas/bool e atualiza self.df_base.

        - Usa self.num_cols se definida; caso contrário detecta automaticamente.
        - Ajusta scaler apenas em modo train para evitar leakage.
        """
        if self.df_base is None or self.df_base.empty:
            print("⚠️ df_base vazio. Chame load_data() primeiro.")
            return self.df_base

        df = self.df_base.copy()

        # Determina colunas numéricas
        cols = self.num_cols

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
            # Removendo _noise das colunas a serem normalizadas
            true_cols = [c for c in cols if not c.endswith("_noise")]
            noise_cols = [c for c in cols if c.endswith("_noise")]
            df[true_cols] = scaler.transform(df[true_cols])

            # fazendo normalização manual para colunas com _noise baseado na métricas do scaler
            for col in noise_cols:
                base_col = col.replace("_noise", "")
                if base_col in self.norm_objects[normalization_method]['num_cols']:
                    mean = scaler.mean_[self.norm_objects[normalization_method]['num_cols'].index(base_col)]
                    scale = scaler.scale_[self.norm_objects[normalization_method]['num_cols'].index(base_col)]
                    df[col] = (df[col] - mean) / scale

        # Caso contrário: cria e ajusta novo scaler
        else:
            scaler = scaler_class()
            df[cols] = scaler.fit_transform(df[cols])
            self.norm_objects[normalization_method] = {
                'num_cols': cols,
                'scaler': scaler
            }

        self.df_base = df
        return self.df_base
    
    def normalize_splits(self, normalization_method: str = 'minmax', train: bool = False) -> dict:
        """Normaliza os conjuntos de treino, validação e teste usando self.num_cols.

        - Detecta colunas numéricas na primeira chamada se self.num_cols estiver vazia.
        - Ajusta scaler apenas no split 'train' quando train=True.
        """
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
            normalized_df = self.normalize(normalization_method=normalization_method, train=train)
            normalized_splits[split_name] = normalized_df
        self.splits = normalized_splits
        return normalized_splits

    def denormalize(self, normalization_method: str = 'minmax', df: pd.DataFrame = None) -> pd.DataFrame:
        """Reverte normalização usando metadados salvos."""
        if df is None or df.empty:
            print("df_base vazio. Nada para denormalizar.")
            return df
        info = self.norm_objects.get(normalization_method)
        if not info:
            print(f"Nenhum scaler salvo para o método '{normalization_method}'.")
            return df
        base_cols = info['num_cols']          # ex: ["A", "B", "C"]

        selected = [
            c
            for c in df.columns
            if any(c == base or c.startswith(base + "_") for base in base_cols)
]
        scaler = info['scaler']
        try:
            df[selected] = scaler.inverse_transform(df[selected].values)
        except Exception as e:
            print(f"Falha ao denormalizar colunas {selected}: {e}")
            return df
        return df

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

    def split_train_val_test(self, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15, time_col: str = 'datetime', dataset_keep: Optional[List[str]] = ["train", "val", "test"]) -> Optional[dict]:
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
        # Filtra splits conforme dataset_keep
        splits = {k: v for k, v in splits.items() if k in dataset_keep}
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
    