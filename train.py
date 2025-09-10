import cloudpickle
import numpy as np
import os
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, MinMaxScaler, TargetEncoder,OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin, clone, TransformerMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


set_config(transform_output="pandas")

# Winsorizer
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, p=0.03):
        self.p = p

    def fit(self, X, y=None):
        self.bounds_ = {}
        for col in X.select_dtypes(include='number').columns:
            Q1 = X[col].quantile(self.p)
            Q3 = X[col].quantile(1 - self.p)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_out = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            X_out[col] = np.clip(X_out[col], lower, upper)
        return X_out

def obtenerXy(X):
  X = X.copy()
  y = X.target
  X = X.drop(columns=["target"])
  return X, y


def medianas_c(df, drop_columns,ceros):
  for col in drop_columns:
    df = df.drop(columns=[col])
  medianas=df.select_dtypes(include='number')
  for exc in ceros:
    medianas=medianas.drop(columns=[exc])
  medianas=medianas.columns
  return medianas

def Map():
    continent_map = {
    'SPA': 'Europe', 'DEU': 'Europe', 'GBR': 'Europe', 'FRA': 'Europe',
    'ITA': 'Europe', 'SWE': 'Europe', 'POL': 'Europe', 'NLD': 'Europe',
    'CHE': 'Europe', 'AUT': 'Europe', 'IRL': 'Europe', 'BGR': 'Europe',
    'CZE': 'Europe', 'ROU': 'Europe', 'SVN': 'Europe', 'HRV': 'Europe',
    'DNK': 'Europe', 'RUS': 'Europe', 'UKR': 'Europe', 'FIN': 'Europe',
    'LUX': 'Europe', 'GRC': 'Europe', 'EST': 'Europe', 'LVA': 'Europe',
    'SVK': 'Europe', 'BIH': 'Europe', 'LTU': 'Europe', 'LIE': 'Europe',
    'SRB': 'Europe', 'CYP': 'Europe', 'MLT': 'Europe', 'ISL': 'Europe',
    'ALB': 'Europe', 'MKD': 'Europe', 'MNE': 'Europe', 'JEY': 'Europe',
    'MCO': 'Europe', 'GIB': 'Europe', 'JEY': 'Europe', 'MCO': 'Europe',
    'GIB': 'Europe', 'POR': 'Europe',

    'USA': 'America', 'MEX': 'America', 'BRA': 'America', 'DOM': 'America',
    'COL': 'America', 'ARG': 'America', 'CRI': 'America', 'PER': 'America',
    'CHL': 'America', 'ECU': 'America', 'VEN': 'America', 'URY': 'America',
    'CUB': 'America', 'PRI': 'America', 'SLV': 'America', 'HND': 'America',
    'SUR': 'America', 'BOL': 'America', 'PRY': 'America', 'JAM': 'America',
    'LCA': 'America', 'GUY': 'America', 'BAR': 'America',

    'JPN': 'Asia', 'TUR': 'Asia', 'ISR': 'Asia', 'IRQ': 'Asia', 'KOR': 'Asia',
    'SAU': 'Asia', 'IND': 'Asia', 'CHN': 'Asia', 'IRN': 'Asia', 'KAZ': 'Asia',
    'IDN': 'Asia', 'KWT': 'Asia', 'LBN': 'Asia', 'TWN': 'Asia', 'JOR': 'Asia',
    'LKA': 'Asia', 'ARM': 'Asia', 'MAC': 'Asia', 'ARE': 'Asia', 'HKG': 'Asia',
    'PHL': 'Asia', 'SGP': 'Asia', 'THA': 'Asia', 'VNM': 'Asia', 'PAK': 'Asia',
    'UZB': 'Asia', 'OMN': 'Asia', 'LAO': 'Asia', 'KHM': 'Asia', 'BGD': 'Asia',
    'QAT': 'Asia', 'SYR': 'Asia',

    'DZA': 'Africa', 'AGO': 'Africa', 'CMR': 'Africa', 'EGY': 'Africa',
    'MAR': 'Africa', 'GNB': 'Africa', 'MOZ': 'Africa', 'ZAF': 'Africa',
    'LBY': 'Africa', 'ETH': 'Africa', 'NGA': 'Africa', 'SEN': 'Africa',
    'CPV': 'Africa', 'CIV': 'Africa', 'BEN': 'Africa', 'TGO': 'Africa',
    'GHA': 'Africa', 'SLE': 'Africa', 'STP': 'Africa', 'RWA': 'Africa',
    'GAB': 'Africa', 'MUS': 'Africa', 'MLI': 'Africa', 'UGA': 'Africa',
    'ZMB': 'Africa', 'SDN': 'Africa', 'ZWE': 'Africa', 'MWI': 'Africa',

    'AUS': 'Oceania', 'NZL': 'Oceania', 'PYF': 'Oceania', 'PLW': 'Oceania',
    'NCL': 'Oceania',

    'ATA': 'Other', 'TMP': 'Other', 'CYM': 'Other', 'AND': 'Other', 'MDV': 'Other',

    'HUN': 'Europe',
    'BEL': 'Europe',
    'CN': 'Asia',
    'BLR': 'Europe',
    'NOR': 'Europe',
    'MYS': 'Asia',
    'GEO': 'Asia',
    'TUN': 'Africa',
    'AZE': 'Asia',
    'CAF': 'Africa',
    'KEN': 'Africa',
    'ABW': 'Caribbean',
    'BRB': 'Caribbean',
    'SYC': 'Africa',
    'VGB': 'Caribbean',
    'NPL': 'Asia'
    }

    return continent_map


def get_pipeline():
    imputacion,outliers,p,drop_columns="mean","otro",0.03,["hotel_id"]
    ceros=["required_car_parking_spaces","special_requests"]  
    X, y = get_X_y()  
    medianas=medianas_c(X,drop_columns,ceros)

    # DEFINICIÓN DE LA PIPELIENE + HIPERPARÁMETROS SELECCIONADOS
    col_remover = ColumnTransformer(
        transformers=[("drop", "drop", drop_columns)],
        remainder="passthrough", 
        force_int_remainder_cols=False,
        verbose_feature_names_out=False 
        )


    # Transformaciones específicas por tipo de dato

    target_encoder = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=pd.NA, strategy='most_frequent')),
    ('target_enc', TargetEncoder())
    ])

    # 2. OrdinalEncoder para room_type
    ordinal_encoder = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=pd.NA, strategy='most_frequent')),
        ('ordinal_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 3. OneHotEncoder para todas las demás categóricas (excepto country_x y room_type)
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=pd.NA, strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore'))
    ])

    # Numéricas no-zero
    num_transformer_no_zero = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=pd.NA, strategy=imputacion)),
        ('scaler', StandardScaler())
    ])

    # Numéricas zero
    num_transformer_zero = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value=0)),
        ('scaler', MinMaxScaler(clip=True)) # Cuando imputamos 0, no tiene mucho sentido hacer standard scaler, porque perdemos la referencia del 0
    ])

    transformer = ColumnTransformer(
        transformers=[
            ('num_zero', num_transformer_zero, ceros),
            ('num_no_zero', num_transformer_no_zero, medianas),
            ('target_country', target_encoder, selector(pattern="^country_x$")),
            ('ordinal_room', ordinal_encoder, selector(pattern="^room_type$")),
            ('onehot_cat', cat_transformer, selector(dtype_include="object", pattern="^(?!.*(country_x|room_type)).*$")),
        ],
        verbose_feature_names_out=False # Introduce un prefijo al nuevo nombre de columna para saber de donde viene
    )

    preprocess_pipeline = Pipeline(steps=[
        ('col', col_remover),
        ('cast', FunctionTransformer(lambda x: x.infer_objects(), validate=False)),
        ('outlier_imputer', Winsorizer(p=p)),
        ('transformer', transformer),
        ('variance_threshold', VarianceThreshold())
        ])

    preprocess_pipeline.fit_transform(X,y)
    pipe = ImbPipeline(steps=[
        *preprocess_pipeline.steps,
        ("sampler", SMOTE(k_neighbors=5)), 
        ("clf", LGBMClassifier(
            colsample_bytree=0.8058265802441404,
            learning_rate=0.012119891565915222,
            max_depth=11,
            min_child_samples=21,
            n_estimators=158,
            num_leaves=189,
            subsample=0.5233328316068078,
            random_state=42
        ))
    ])

    return pipe

def get_X_y():
  dfb = pd.read_csv(os.environ["BOOKINGS_PATH"])
  dfh = pd.read_csv(os.environ["HOTELS_PATH"])

  dfb=dfb[dfb["reservation_status"]!="Booked"]
  dfb=dfb[dfb["stay_nights"]>0]

  for fecha in ["booking_date","arrival_date","reservation_status_date"]:
    dfb[fecha] = pd.to_datetime(dfb[fecha])
  dfb["rate_day"] = np.where(dfb["stay_nights"] == 0, dfb["rate"] / 1, dfb["rate"] / dfb["stay_nights"])
  dfb = dfb.drop(columns=['rate'])
  dfb["dif_date"]=(dfb.arrival_date-dfb.booking_date).dt.days

  dfb = dfb[dfb['reservation_status'] != 'Booked']
  dfb['target'] = (dfb['reservation_status'] == 'Canceled') & ((dfb['arrival_date'] - dfb['reservation_status_date']).dt.days < 30)
  dfb = dfb.drop(columns=['reservation_status'])
  df = dfb.merge(dfh, on='hotel_id', how='left')

  continent_map=Map()
  df["local"] = df["country_x"] == df["country_y"]
  df["continent"] = df["country_x"].map(continent_map)
  df["arrival_weekday"] = df["arrival_date"].dt.weekday
  df["arrival_day"] = df["arrival_date"].dt.day
  df["arrival_year"] = df["arrival_date"].dt.year

  df[f"arrival_month_sen"] = np.sin(2 * np.pi * df["arrival_date"].dt.month / 12)
  df[f"arrival_month_cos"] = np.cos(2 * np.pi * df["arrival_date"].dt.month / 12)
  df[f"arrival_weekday_sen"] = np.sin(2 * np.pi * df["arrival_date"].dt.month / 7)
  df[f"arrival_weekday_cos"] = np.cos(2 * np.pi * df["arrival_date"].dt.month / 7)

  df = df.drop(columns=["booking_date","reservation_status_date","arrival_date","arrival_weekday"])

  X,y=obtenerXy(df)
  
  return X,y


def save_pipeline(pipe):
    # AQUI SE SERIALIZA EL MODELO. PUEDE SER TAMBIÉN CON JOBLIB
    with open(os.environ["MODEL_PATH"], mode="wb") as f:
        cloudpickle.dump(pipe, f)


if __name__ == "__main__":
    X, y = get_X_y()
    pipe = get_pipeline()
    pipe.fit(X, y)
    save_pipeline(pipe)