import streamlit as st
import yfinance as yf
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
import exchange_calendars as xcals
import numpy as np

def formatea_precios_yahoo(bmks_rv, fecha):
    precios_bmks_yahoo_df = yf.download(bmks_rv, start=datetime(year=fecha.year - 1, month=1, day=1).strftime("%Y-%m-%d"))
    precios_bmks_yahoo_df = precios_bmks_yahoo_df.xs(key="Close", axis=1, level=0)
    precios_bmks_yahoo_df.reset_index(inplace=True)
    precios_bmks_yahoo_df.rename(columns={"Date":"Fecha", "^MXX":"IPC", "^SPESG":"SPESG_USD", "^SPGSCI":"SPGSCI_USD", "^GSPC":"S&P_USD", "^NDX":"NDX_USD"}, inplace=True)
    precios_bmks_yahoo_df["Fecha"] = pd.to_datetime(precios_bmks_yahoo_df["Fecha"])

    return precios_bmks_yahoo_df

def formatea_precios_spot():
    spot_df = pd.read_csv("./ArchivosRendimientos/Benchmarks/Historico_SPOT.csv")
    spot_df.rename(columns={"FECHA":"Fecha", "PRECIO SUCIO":"Spot"}, inplace=True)
    spot_df = spot_df[["Fecha", "Spot"]]
    spot_df["Fecha"] = pd.to_datetime(spot_df["Fecha"])

    return spot_df

def formatea_precios_isimp():
    precios_bmks_isimp_df = pd.read_excel("./ArchivosPeergroups/Benchmarks/Historico_ISIMP.xlsx", skiprows=2, skipfooter=4)
    precios_bmks_isimp_df["Fecha"] = pd.to_datetime(precios_bmks_isimp_df["Fecha"], format="%d/%m/%Y")
    precios_bmks_isimp_df.rename(columns={"Índice":"ISIMP"}, inplace=True)
    precios_bmks_isimp_df = precios_bmks_isimp_df[["Fecha", "ISIMP"]]

    return precios_bmks_isimp_df

def formatea_precios_acwi():
    precios_bmks_acwi_df = pd.read_csv("./ArchivosPeergroups/Benchmarks/Historico_ACWI.csv")
    precios_bmks_acwi_df.rename(columns={"FECHA":"Fecha", "PRECIO SUCIO":"ACWI"}, inplace=True)
    precios_bmks_acwi_df["Fecha"] = pd.to_datetime(precios_bmks_acwi_df["Fecha"], format="%Y-%m-%d")
    precios_bmks_acwi_df = precios_bmks_acwi_df[["Fecha", "ACWI"]]

    return precios_bmks_acwi_df

def formatea_precios_bmks_valmer():
    precios_bmks_valmer_df = pd.read_csv("./ArchivosPeergroups/Benchmarks/Benchmarks_SP_Historico_MD.csv")
    precios_bmks_valmer_df.rename(columns={"FECHA":"Fecha"}, inplace=True)
    precios_bmks_valmer_df["Fecha"] = pd.to_datetime(precios_bmks_valmer_df["Fecha"], format="%Y%m%d")
    precios_bmks_valmer_df.drop(columns=[col for col in precios_bmks_valmer_df.columns if "Unnamed" in col], inplace=True)

    return precios_bmks_valmer_df

def formatea_precios_bmks(fecha):
    spot_df = formatea_precios_spot()
    precios_bmks_yahoo_df = formatea_precios_yahoo(bmks_rv, fecha)
    precios_bmks_df = pd.merge(precios_bmks_yahoo_df, spot_df, on="Fecha", how="left")
    precios_bmks_df["SPESG"] = precios_bmks_df["SPESG_USD"] * precios_bmks_df["Spot"]
    precios_bmks_df["SPGSCI"] = precios_bmks_df["SPGSCI_USD"] * precios_bmks_df["Spot"]
    precios_bmks_df["S&P"] = precios_bmks_df["S&P_USD"] * precios_bmks_df["Spot"]
    precios_bmks_df["NDX"] = precios_bmks_df["NDX_USD"] * precios_bmks_df["Spot"]
    precios_bmks_df.drop(columns=["SPESG_USD", "SPGSCI_USD", "S&P_USD", "NDX_USD"], inplace=True)
    precios_bmks_isimp_df = formatea_precios_isimp()
    precios_bmks_df = pd.merge(precios_bmks_df, precios_bmks_isimp_df, on="Fecha", how="left")
    precios_bmks_acwi_df = formatea_precios_acwi()
    precios_bmks_df = pd.merge(precios_bmks_df, precios_bmks_acwi_df, on="Fecha", how="left")
    precios_bmks_valmer_df = formatea_precios_bmks_valmer()
    precios_bmks_df = pd.merge(precios_bmks_df, precios_bmks_valmer_df, on="Fecha", how="left")
    precios_bmks_df.set_index("Fecha", inplace=True)

    return precios_bmks_df

def infer_calendar(dates):
    """
    Infer a calendar as pandas DateOffset from a list of dates.
    Parameters
    ----------
    dates : array-like (1-dimensional) or pd.DatetimeIndex
        The dates you want to build a calendar from
    Returns
    -------
    calendar : pd.DateOffset (CustomBusinessDay)
    """
    dates = pd.DatetimeIndex(dates)

    traded_weekdays = []
    holidays = []

    days_of_the_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day, day_str in enumerate(days_of_the_week):

        weekday_mask = (dates.dayofweek == day)

        # keep only days of the week that are present
        if not weekday_mask.any():
            continue
        traded_weekdays.append(day_str)

        # look for holidays
        used_weekdays = dates[weekday_mask].normalize()
        all_weekdays = pd.date_range(dates.min(), dates.max(),
                                     freq=CustomBusinessDay(weekmask=day_str)
                                     ).normalize()
        _holidays = all_weekdays.difference(used_weekdays)
        _holidays = [timestamp.date() for timestamp in _holidays]
        holidays.extend(_holidays)

    traded_weekdays = ' '.join(traded_weekdays)
    return CustomBusinessDay(weekmask=traded_weekdays, holidays=holidays)

def calcula_fechas_exactas_iniciales(fecha):
    if fecha.month == 1:
        year = fecha.year - 1
        month = 12
    else:
        year = fecha.year
        month = fecha.month - 1

    fechas_exactas_iniciales = {
        "MTD": datetime(year=year, month=month, day=calendar.monthrange(year, month)[1]),
        "YTD": datetime(year=fecha.year - 1, month=12, day=calendar.monthrange(year=fecha.year - 1, month=12)[1]),
        "12 Meses": fecha - timedelta(days=366 if calendar.isleap(year) else 365),
        "30D": fecha - timedelta(days=30),
        "90D": fecha - timedelta(days=90),
        "180D": fecha - timedelta(days=180)
    }

    return fechas_exactas_iniciales

def calcula_fecha_habil_proxima_anterior(fecha_exacta, fechas_bmv, bmv_offset):
    return (fecha_exacta - bmv_offset).to_pydatetime() if fecha_exacta not in fechas_bmv else fecha_exacta

def calcula_fecha_habil_proxima_posterior(fecha_exacta, fechas_bmv, bmv_offset):
    return (fecha_exacta + bmv_offset).to_pydatetime() if fecha_exacta not in fechas_bmv else fecha_exacta

def calcula_fechas_habiles_iniciales(fechas_exactas_iniciales, fechas_bmv, bmv_offset, tipo="Deuda"):
    if tipo == "Deuda":
        fechas_habiles_iniciales = {
            ventana:calcula_fecha_habil_proxima_anterior(fechas_exactas_iniciales[ventana], fechas_bmv, bmv_offset) for ventana in fechas_exactas_iniciales.keys()
        }
    
    else:
        fechas_habiles_iniciales = {
            ventana:calcula_fecha_habil_proxima_posterior(fechas_exactas_iniciales[ventana] + timedelta(days=1 if ventana in ["MTD", "YTD"] else 0), fechas_bmv, bmv_offset) for ventana in fechas_exactas_iniciales.keys()
        }

    return fechas_habiles_iniciales

fondo2benchmark = {
    "VECTUSA":{
        "Benchmarks":[
            "SPESG"
        ],
        "Pesos":[1.0]
    },
    "VECTCOB":{
        "Benchmarks":[
            "Spot"
        ],
        "Pesos":[1.0]
    },
    "VECTCOR":{
        "Benchmarks":[
            "ISIMP"
        ],
        "Pesos":[1.0]
    },
    "VECTUSD":{
        "Benchmarks":[
            "Spot"
        ],
        "Pesos":[1.0]
    },
    "VECTFI":{
        "Benchmarks":[
            "S&P/BMV Sovereign CETES 7+ Day Bond Index",
            "S&P/BMV Sovereign MBONOS 1-3 Year Bond Index"
        ],
        "Pesos":[0.5, 0.5]
    },
    "VECTIND":{
        "Benchmarks":[
            "IPC"
        ],
        "Pesos":[1.0]
    },
    "VECTMD":{
        "Benchmarks":[
            "S&P/BMV Sovereign CETES 28 Day Bond Index"
        ],
        "Pesos":[1.0]
    },
    "DYNAMIC":{
        "Benchmarks":[],
        "Pesos":[]
    },
    "VECTPA":{
        "Benchmarks":[
            "IPC"
        ],
        "Pesos":[1.0]
    },
    "VECTPRE":{
        "Benchmarks":[
            "S&P/BMV Sovereign CETES 28 Day Bond Index"
        ],
        "Pesos":[1.0]
    },
    "VECTPZO":{
        "Benchmarks":[
            "S&P/BMV Sovereign MBONOS 3-5 Year Bond Index"
        ],
        "Pesos":[1.0]
    },
    "VECTRF":{
        "Benchmarks":[
            "S&P/BMV Sovereign CETES 28 Day Bond Index"
        ],
        "Pesos":[1.0]
    },
    "VECTSIC":{
        "Benchmarks":[
            "ACWI"
        ],
        "Pesos":[1.0]
    },
    "INCOME":{
        "Benchmarks":[
            "S&P/BMV Sovereign CETES 28 Day Bond Index", 
            "S&P/BMV Sovereign CETES 7+ Day Bond Index", 
            "S&P/BMV Sovereign MBONOS 1-3 Year Bond Index"
        ],
        "Pesos":[0.4, 0.3, 0.3]
    },
    "EQUITY":{
        "Benchmarks":[
            "ACWI", 
            "IPC"
        ],
        "Pesos":[0.5, 0.5]
    },
    "BALANCE":{
        "Benchmarks":[
            "S&P/BMV Sovereign CETES 28 Day Bond Index", 
            "IPC", 
            "ACWI"
        ],
        "Pesos":[0.7, 0.15, 0.15]
    },
    "VECTTR":{
        "Benchmarks":[
            "S&P/BMV Sovereign UDIBONOS 1-3 Year Bond Index",
            "S&P/BMV Sovereign UDIBONOS 5-10 Year Bond Index"
        ],
        "Pesos":[0.4, 0.6]
    },
    "VECTMIX":{
        "Benchmarks":[
            "S&P", "IPC"
        ],
        "Pesos":[0.5, 0.5]
    },
    "COMMODQ":{
        "Benchmarks":[
            "SPGSCI"
        ],
        "Pesos":[1.0]
    },
    "MXRATES":{
        "Benchmarks":[],
        "Pesos":[]
    },
    "NEXTGEN":{
        "Benchmarks":[
            "NDX"
        ],
        "Pesos":[1.0]
    }
}

bmks_rv = ["^MXX", "^SPESG", "^SPGSCI", "^GSPC", "^NDX"]

fecha = datetime.today()
fecha = datetime(year=fecha.year, month=fecha.month, day=fecha.day)

xmex = xcals.get_calendar("XMEX")
fechas_bmv = sorted(xmex.sessions_in_range(
    start=(datetime.today() - relativedelta(years=20) + timedelta(days=3)).strftime("%Y-%m-%d"), 
    end=datetime.today().strftime("%Y-%m-%d")
).to_pydatetime())
fechas_bmv.remove(datetime(2024, 10, 1, 0, 0))

bmv_offset = infer_calendar(fechas_bmv)

fechas_exactas_iniciales_rf = calcula_fechas_exactas_iniciales(fecha)
fechas_habiles_iniciales_rf = calcula_fechas_habiles_iniciales(fechas_exactas_iniciales_rf, fechas_bmv, bmv_offset, tipo="Deuda")

fechas_exactas_iniciales_rv = calcula_fechas_exactas_iniciales((fecha - bmv_offset).to_pydatetime())
fechas_habiles_iniciales_rv = calcula_fechas_habiles_iniciales(fechas_exactas_iniciales_rv, fechas_bmv, bmv_offset, tipo="RV")

precios_bmks_df = formatea_precios_bmks(fecha)

st.write(precios_bmks_df)
st.write(fechas_habiles_iniciales_rf)
st.write(fechas_habiles_iniciales_rv)

rendimientos_bmks_df = pd.DataFrame()
for ventana in fechas_habiles_iniciales_rv.keys():
    rendimientos_bmk_ventana = []
    for fondo in fondo2benchmark.keys():
        fecha_inicial = fechas_habiles_iniciales_rv[ventana]
        fecha_final = (fecha - bmv_offset).to_pydatetime()

        bmks = fondo2benchmark[fondo]["Benchmarks"]
        pesos = fondo2benchmark[fondo]["Pesos"]
        rendimiento_bmk = ((precios_bmks_df[bmks].loc[fecha_final]/precios_bmks_df[bmks].loc[fecha_inicial] - 1) * pesos).sum()
        if len(bmks) == 0:
            rendimiento_bmk = np.nan

        rendimientos_bmk_ventana.append(rendimiento_bmk)

    rendimientos_bmk_ventana_df = pd.DataFrame({ventana:rendimientos_bmk_ventana}, index=fondo2benchmark.keys())
    rendimientos_bmks_df = pd.concat([rendimientos_bmks_df, rendimientos_bmk_ventana_df], axis=1)


st.write(rendimientos_bmks_df)