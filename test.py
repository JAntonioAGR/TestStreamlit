import streamlit as st
import yfinance as yf

bmks_rv = [
    "^MXX", "^SPESG"
]
df = yf.download(bmks_rv, start='2020-01-01')

st.write(df)