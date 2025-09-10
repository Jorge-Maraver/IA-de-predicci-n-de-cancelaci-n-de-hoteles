import os
import numpy as np
import cloudpickle
import pandas as pd
from sklearn import set_config

set_config(transform_output="pandas")
bookings = pd.read_csv(os.environ["BOOKINGS_PATH"])
hotels = pd.read_csv(os.environ["HOTELS_PATH"])

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


def get_X():
  dfb = pd.read_csv(os.environ["INFERENCE_DATA_PATH"])
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

  X=df.drop(columns=["target"])
  
  return X

def get_pipeline():
    with open(os.environ["MODEL_PATH"], mode="rb") as f:
        pipe = cloudpickle.load(f)

    return pipe

def get_predictions(pipe):

    y_pred = pipe.predict(X)

    X["prediction"] = y_pred
    return X[["prediction"]]#.to_csv("output_predictions.csv", index=None)

if __name__ == "__main__":
    X = get_X()
    pipe = get_pipeline()
    preds = get_predictions(pipe)

    # NO CAMBIAR LA RUTA DE SALIDA NI EL FORMATO. UNA ÚNICA COLUMNA CON LAS PREDICCIONES 0/1
    preds.to_csv("output_predictions.csv")
