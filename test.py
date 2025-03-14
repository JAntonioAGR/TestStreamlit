import streamlit as st
import yfinance as yf
from datetime import datetime
import pandas as pd

fecha = datetime.today()
fecha = datetime(year=fecha.year, month=fecha.month, day=fecha.day)

fondo2benchmark = {
    "VECTUSA":{
        "Benchmarks":["SPESG"],
        "Pesos":[1.0]
    },
    "VECTCOB":{
        "Benchmarks":["Spot Valmer"],
        "Pesos":[1.0]
    },
    "VECTCOR":{
        "Benchmarks":["ISIMP"],
        "Pesos":[1.0]
    },
    "VECTUSD":{
        "Benchmarks":["Spot Valmer"],
        "Pesos":[1.0]
    },
    "VECTFI":{
        "Benchmarks":["CETES 7+", "MBONOS 1-3"],
        "Pesos":[0.5, 0.5]
    },
    "VECTIND":{
        "Benchmarks":["IPC"],
        "Pesos":[1.0]
    },
    "VECTMD":{
        "Benchmarks":["CETES 28 AN"],
        "Pesos":[1.0]
    },
    "DYNAMIC":{
        "Benchmarks":[],
        "Pesos":[]
    },
    "VECTPA":{
        "Benchmarks":["IPC"],
        "Pesos":[1.0]
    },
    "VECTPRE":{
        "Benchmarks":["CETES 28 AN"],
        "Pesos":[1.0]
    },
    "VECTPZO":{
        "Benchmarks":["MBONOS 3-5"],
        "Pesos":[1.0]
    },
    "VECTRF":{
        "Benchmarks":["CETES 28 AN"],
        "Pesos":[1.0]
    },
    "VECTSIC":{
        "Benchmarks":["ACWI"],
        "Pesos":[1.0]
    },
    "INCOME":{
        "Benchmarks":["CETES 28 AN", "CETES 7+", "MBONOS 1-3"],
        "Pesos":[0.4, 0.3, 0.3]
    },
    "EQUITY":{
        "Benchmarks":["ACWI", "IPC"],
        "Pesos":[0.5, 0.5]
    },
    "BALANCE":{
        "Benchmarks":["CETES 28 DIR", "IPC", "ACWI"],
        "Pesos":[0.7, 0.15, 0.15]
    },
    "VECTTR":{
        "Benchmarks":["VECTTR"],
        "Pesos":[1.0]
    },
    "VECTMIX":{
        "Benchmarks":["S&P 500", "IPC"],
        "Pesos":[0.5, 0.5]
    },
    "COMMODQ":{
        "Benchmarks":["COMMODQ"],
        "Pesos":[1.0]
    },
    "MXRATES":{
        "Benchmarks":[],
        "Pesos":[]
    },
    "NEXTGEN":{
        "Benchmarks":["NASDAQ"],
        "Pesos":[1.0]
    }
}

bmks_rv = [
    "^MXX", "^SPESG", "^SPGSCI"
]
precios_bmks_yahoo_df = yf.download(bmks_rv, start=datetime(year=fecha.year - 1, month=1, day=1).strftime("%Y-%m-%d"))
precios_bmks_yahoo_df = precios_bmks_yahoo_df.xs(key="Close", axis=1, level=0)
precios_bmks_yahoo_df.reset_index(inplace=True)
precios_bmks_yahoo_df.rename(columns={"Date":"Fecha", "^MXX":"IPC", "^SPESG":"SPESG_USD", "^SPGSCI":"SPGSCI_USD"}, inplace=True)
precios_bmks_yahoo_df["Fecha"] = pd.to_datetime(precios_bmks_yahoo_df["Fecha"])

spot_df = pd.read_csv("./ArchivosRendimientos/Benchmarks/Historico_SPOT.csv")
spot_df.rename(columns={"FECHA":"Fecha", "PRECIO SUCIO":"Spot"}, inplace=True)
spot_df = spot_df[["Fecha", "Spot"]]
spot_df["Fecha"] = pd.to_datetime(spot_df["Fecha"])

precios_bmks_yahoo_df = pd.merge(precios_bmks_yahoo_df, spot_df, on="Fecha", how="left")
precios_bmks_yahoo_df["SPESG"] = precios_bmks_yahoo_df["SPESG_USD"] * precios_bmks_yahoo_df["Spot"]
precios_bmks_yahoo_df["SPGSCI"] = precios_bmks_yahoo_df["SPGSCI_USD"] * precios_bmks_yahoo_df["Spot"]

precios_bmks_isimp_df = pd.read_excel("./ArchivosPeergroups/Benchmarks/Historico_ISIMP.xlsx", skiprows=2, skipfooter=4)
precios_bmks_isimp_df["Fecha"] = pd.to_datetime(precios_bmks_isimp_df["Fecha"], format="%d/%m/%Y")
precios_bmks_isimp_df.rename(columns={"Índice":"ISIMP"}, inplace=True)
precios_bmks_isimp_df = precios_bmks_isimp_df[["Fecha", "ISIMP"]]

precios_bmks_yahoo_df = pd.merge(precios_bmks_yahoo_df, precios_bmks_isimp_df, on="Fecha", how="left")

precios_bmks_acwi_df = pd.read_csv("./ArchivosPeergroups/Benchmarks/Historico_ACWI.csv")
precios_bmks_acwi_df.rename(columns={"FECHA":"Fecha", "PRECIO SUCIO":"ACWI"}, inplace=True)
precios_bmks_acwi_df["Fecha"] = pd.to_datetime(precios_bmks_acwi_df["Fecha"], format="%Y-%m-%d")
precios_bmks_acwi_df = precios_bmks_acwi_df[["Fecha", "ACWI"]]

precios_bmks_yahoo_df = pd.merge(precios_bmks_yahoo_df, precios_bmks_acwi_df, on="Fecha", how="left")

st.write(precios_bmks_yahoo_df)