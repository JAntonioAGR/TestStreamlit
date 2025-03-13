import streamlit as st
import yfinance as yf
from datetime import datetime

fecha = datetime.today()
fecha = datetime(year=fecha.year, month=fecha.month, day=fecha.day)

bmks_rv = [
    "^MXX", "^SPESG", "^SPGSCI"
]
df = yf.download(bmks_rv, start=datetime(year=fecha.year - 1, month=1, day=1).strftime("%Y-%m-%d"))

st.write(df)