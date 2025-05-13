import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
import exchange_calendars as xcals
import numpy as np
import os
import time
import io
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# import general_function.general_func as gf

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
    precios_bmks_df = pd.merge(precios_bmks_yahoo_df, spot_df, on="Fecha", how="outer")
    precios_bmks_df["SPESG"] = precios_bmks_df["SPESG_USD"] * precios_bmks_df["Spot"]
    precios_bmks_df["SPGSCI"] = precios_bmks_df["SPGSCI_USD"] * precios_bmks_df["Spot"]
    precios_bmks_df["S&P"] = precios_bmks_df["S&P_USD"] * precios_bmks_df["Spot"]
    precios_bmks_df["NDX"] = precios_bmks_df["NDX_USD"] * precios_bmks_df["Spot"]
    precios_bmks_df.drop(columns=["SPESG_USD", "SPGSCI_USD", "S&P_USD", "NDX_USD"], inplace=True)
    precios_bmks_isimp_df = formatea_precios_isimp()
    precios_bmks_df = pd.merge(precios_bmks_df, precios_bmks_isimp_df, on="Fecha", how="outer")
    precios_bmks_acwi_df = formatea_precios_acwi()
    precios_bmks_df = pd.merge(precios_bmks_df, precios_bmks_acwi_df, on="Fecha", how="outer")
    precios_bmks_valmer_df = formatea_precios_bmks_valmer()
    precios_bmks_df = pd.merge(precios_bmks_df, precios_bmks_valmer_df, on="Fecha", how="outer")
    precios_bmks_df.set_index("Fecha", inplace=True)
    precios_bmks_df.sort_index(inplace=True)

    return precios_bmks_df.ffill()

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
            ventana:calcula_fecha_habil_proxima_posterior(fechas_exactas_iniciales[ventana], fechas_bmv, bmv_offset) if ventana in ["MTD", "YTD"] else
            calcula_fecha_habil_proxima_anterior(fechas_exactas_iniciales[ventana], fechas_bmv, bmv_offset) for ventana in fechas_exactas_iniciales.keys()
        }

    return fechas_habiles_iniciales

def formatea_rendimientos_bmk(fecha, precios_bmks_df, fechas_habiles_iniciales_rf, fechas_habiles_iniciales_rv, propiedades_fondos_df, bmv_offset, fondo2benchmark):
    rendimientos_bmks_df = pd.DataFrame()
    for ventana in fechas_habiles_iniciales_rv.keys():
        rendimientos_bmk_ventana = []
        for fondo in fondo2benchmark.keys():
            tipo_fondo = propiedades_fondos_df.loc[propiedades_fondos_df["Fondo"] == fondo, "Tipo"].item()
            fechas_habiles_iniciales = fechas_habiles_iniciales_rv if tipo_fondo == "RV" else fechas_habiles_iniciales_rf

            fecha_inicial = (fechas_habiles_iniciales[ventana] - bmv_offset).to_pydatetime()
            fecha_final = (fecha - bmv_offset).to_pydatetime()

            bmks = fondo2benchmark[fondo]["Benchmarks"]
            pesos = fondo2benchmark[fondo]["Pesos"]
            rendimiento_bmk = ((precios_bmks_df[bmks].loc[fecha_final]/precios_bmks_df[bmks].loc[fecha_inicial] - 1) * pesos).sum()

            if len(bmks) == 0:
                rendimiento_bmk = np.nan
            
            if tipo_fondo == "RF" or fondo in ["BALANCE", "DYNAMIC"]:
                rendimiento_bmk = rendimiento_bmk * 360/(fecha - fechas_habiles_iniciales[ventana]).days

            rendimientos_bmk_ventana.append(rendimiento_bmk)

        rendimientos_bmk_ventana_df = pd.DataFrame({ventana:rendimientos_bmk_ventana}, index=fondo2benchmark.keys())
        rendimientos_bmks_df = pd.concat([rendimientos_bmks_df, rendimientos_bmk_ventana_df], axis=1)

    return rendimientos_bmks_df

def formatea_rendimientos_fondos(fecha, precios_fondos_df, fechas_habiles_iniciales_rf, fechas_habiles_iniciales_rv, propiedades_fondos_df):
    rendimientos_df = pd.DataFrame()
    for tipo in ["RF", "RV"]:
        fondos = propiedades_fondos_df.loc[propiedades_fondos_df["Tipo"] == tipo, "Fondo"].tolist()
        fechas_habiles_iniciales = fechas_habiles_iniciales_rf if tipo == "RF" else fechas_habiles_iniciales_rv
        temp_rendimientos_df = calcula_rendimientos_fondos(precios_fondos_df, fondos, fecha, fechas_habiles_iniciales)
        if temp_rendimientos_df.index.isin(propiedades_fondos_df[~propiedades_fondos_df["Serie"].isin(["XF0", "XF"])]["Fondo"]).any():
            temp_rendimientos_brutos_df = calcula_rendimientos_brutos(temp_rendimientos_df, propiedades_fondos_df, 0.005, fecha, fechas_habiles_iniciales_rf, fechas_habiles_iniciales_rv)
            temp_rendimientos_df.loc[propiedades_fondos_df[~propiedades_fondos_df["Serie"].isin(["XF0", "XF"])]["Fondo"]] = temp_rendimientos_brutos_df.copy()

        rendimientos_df = pd.concat([rendimientos_df, temp_rendimientos_df], axis=0)

    fondos_a_anualizar = propiedades_fondos_df.loc[propiedades_fondos_df["Tipo"] == "RF", "Fondo"].tolist() + ["BALANCE", "DYNAMIC"]
    temp_rendimientos_df = rendimientos_df.loc[fondos_a_anualizar].copy()
    temp_rendimientos_df = calcula_rendimientos_anualizados(temp_rendimientos_df, fecha, fechas_habiles_iniciales_rf)
    rendimientos_df.loc[fondos_a_anualizar] = temp_rendimientos_df.copy()

    return rendimientos_df

def formatea_columna_tabla_rendimientos_MiVector(col):
    temp_col = col.copy()
    if temp_col.dtype == "O":
        temp_col = temp_col.replace("N[/]D[\%]", "", regex=True)
        temp_col = temp_col.replace("%|,", "", regex=True)
        temp_col = temp_col.replace("", np.nan, regex=True)

        if temp_col.isna().sum() != len(temp_col):
            if not temp_col.str.contains(r"[A-Za-z]").any():
                temp_col = temp_col.astype(float)
                temp_col = temp_col/100 if col.name not in ["Precio", "(+)Títulos en Circulación"] else temp_col

    return temp_col

def descarga_rendimientos_MiVector(fechas):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=chrome_options)
    start_url = "https://www.vectoronline.com.mx/servicios/fondos/consultas/precios_rendimientos.php"
    driver.get(start_url)

    rendimientos_MiVector_fechas_df = pd.DataFrame()
    for fecha_dt in fechas:
        fecha_str = fecha_dt.strftime("%Y-%m-%d")
        time.sleep(3)

        driver.find_element(by=By.ID, value="dpFecha").clear()
        driver.find_element(by=By.ID, value="dpFecha").send_keys(fecha_str)
        driver.find_element(by=By.ID, value="btnMostrar").click()

        time.sleep(3)

        tables = pd.read_html(io.StringIO(driver.page_source))

        rendimientos_MiVector_df = []
        for i in range(len(tables)):
            if i % 2 != 0:
                col_names = tables[i - 1].columns
                temp_df = tables[i].rename(columns={i:col_names[i] for i in range(len(col_names))})
                temp_records = temp_df.to_dict(orient="records")
                rendimientos_MiVector_df.extend(temp_records)

        rendimientos_MiVector_df = pd.DataFrame.from_records(rendimientos_MiVector_df)
        rendimientos_MiVector_df["Fecha"] = fecha_str
        rendimientos_MiVector_fechas_df = pd.concat([rendimientos_MiVector_fechas_df, rendimientos_MiVector_df], axis=0, ignore_index=True)

    rendimientos_MiVector_fechas_df["Fecha"] = pd.to_datetime(rendimientos_MiVector_fechas_df["Fecha"], format="%Y-%m-%d")
    rendimientos_MiVector_fechas_df = rendimientos_MiVector_fechas_df.apply(lambda col: formatea_columna_tabla_rendimientos_MiVector(col), axis=0)

    driver.close()

    return rendimientos_MiVector_fechas_df

def calcula_rendimientos_fondos(precios_fondos_df, fondos, fecha, fechas_habiles_iniciales):
    rendimientos_df = pd.DataFrame()
    for ventana in fechas_habiles_iniciales.keys():
        fecha_ventana = fechas_habiles_iniciales[ventana]
        rendimientos_ventana_df = precios_fondos_df.loc[fecha, fondos]/precios_fondos_df.loc[fecha_ventana, fondos] - 1
        # if "VECTRF" in fondos:
        #     st.write(ventana)
        #     st.write(rendimientos_ventana_df)
        rendimientos_ventana_df.name = ventana
        rendimientos_ventana_df = rendimientos_ventana_df.to_frame()
        rendimientos_df = pd.concat([rendimientos_df, rendimientos_ventana_df], axis=1)

    return rendimientos_df

def calcula_rendimientos_anualizados(rendimientos_df, fecha, fechas_habiles_iniciales):
    rendimientos_anualizados_df = rendimientos_df.copy()
    for ventana in rendimientos_anualizados_df:
        dias_transcurridos = (fecha - fechas_habiles_iniciales[ventana]).days
        rendimientos_anualizados_df[ventana] = rendimientos_anualizados_df[ventana] * 360 / dias_transcurridos

    return rendimientos_anualizados_df

def calcula_rendimientos_brutos(rendimientos_df, propiedades_fondos_df, impuesto, fecha, fechas_habiles_iniciales_rf, fechas_habiles_iniciales_rv):
    temp_df = propiedades_fondos_df[~propiedades_fondos_df["Serie"].isin(["XF0", "XF"])].copy()
    temp_df[["Comision", "Factor RF"]] /= 100

    dias_transcurridos_ventanas_df = pd.DataFrame([
            {ventana:(fecha - fechas_habiles_iniciales_rf[ventana]).days for ventana in fechas_habiles_iniciales_rf}|{"Tipo":"RF"},
            {ventana:(fecha - fechas_habiles_iniciales_rv[ventana]).days for ventana in fechas_habiles_iniciales_rv}|{"Tipo":"RV"}
    ])

    temp_df = pd.merge(temp_df, dias_transcurridos_ventanas_df, on="Tipo")
    temp_df.set_index("Fondo", inplace=True)
    temp_df.index.name = None

    comisiones_srs = temp_df["Comision"]
    ajustes_comision_srs = (comisiones_srs * 1.16)/ 360
    ajustes_comision_df = temp_df[list(fechas_habiles_iniciales_rf.keys())].multiply(ajustes_comision_srs, axis=0)

    factores_rf_srs = temp_df["Factor RF"]
    ajustes_impuesto_srs = (factores_rf_srs * impuesto)/360
    ajustes_impuesto_df = temp_df[list(fechas_habiles_iniciales_rf.keys())].multiply(ajustes_impuesto_srs, axis=0)

    rendimientos_brutos_df = rendimientos_df.loc[temp_df.index.tolist()].copy()
    rendimientos_brutos_df = (1 + rendimientos_brutos_df)/(1 - ajustes_comision_df) + ajustes_impuesto_df - 1

    return rendimientos_brutos_df


def grafico_diferencias_rendimiento(rendimientos_df, periodo="MTD", titulo_adicional=""):
    """
    Genera un gráfico de barras horizontales que muestra las diferencias entre los rendimientos
    de los fondos y sus benchmarks para un período específico.
    
    Parámetros:
    rendimientos_df (pandas.DataFrame): DataFrame con los fondos como índice y columnas para rendimientos
    periodo (str): Período a visualizar ("MTD", "YTD", "12 Meses", "30D", "90D", "180D")
    titulo_adicional (str): Texto adicional para el título del gráfico
    
    Retorna:
    fig (plotly.graph_objects.Figure): Figura de Plotly con el gráfico
    """
    # Definir la paleta de colores personalizada
    custom_colors = ['#EC5E2A', '#FF8F66', '#1A3A6C', '#2E5095', '#4268B1', '#5680CE', '#6A98EB']
    
    # Reemplazar valores NaN con 0
    rendimientos_df = rendimientos_df.fillna(0)
    
    # Determinar columnas para fondos y benchmarks
    col_benchmark = f"BMK_{periodo}"
    
    # Verificar si las columnas existen
    if periodo not in rendimientos_df.columns or col_benchmark not in rendimientos_df.columns:
        raise ValueError(f"No se encontraron las columnas {periodo} o {col_benchmark} en el DataFrame")
    
    # Crear una copia del DataFrame para no modificar el original
    df_trabajo = rendimientos_df.copy()
    
    # Caso especial para DYNAMIC: usar 12.0 como benchmark para todas las columnas
    if 'DYNAMIC' in df_trabajo.index:
        for col in df_trabajo.columns:
            if col.startswith('BMK_'):
                df_trabajo.loc['DYNAMIC', col] = 12.0
    
    # Calcular diferencias
    df_trabajo['Diferencia'] = df_trabajo[periodo] - df_trabajo[col_benchmark]
    
    # Dividir en overweight y underweight
    df_trabajo['Underweight'] = df_trabajo['Diferencia'].copy()
    df_trabajo['Overweight'] = df_trabajo['Diferencia'].copy()
    df_trabajo.loc[df_trabajo['Underweight'] > 0, 'Underweight'] = 0
    df_trabajo.loc[df_trabajo['Overweight'] < 0, 'Overweight'] = 0
    
    # Ordenar por diferencia
    df_trabajo = df_trabajo.sort_values(by='Diferencia')
    
    # Crear el gráfico de barras horizontales apiladas en Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_trabajo.index,  # Usar el índice como nombres de fondos
        x=df_trabajo['Underweight'],
        orientation='h',
        marker=dict(
            color=custom_colors[2],
            line=dict(color='white', width=1)  # Agregar borde blanco
        ),
        name='Underperformance'
    ))
    
    fig.add_trace(go.Bar(
        y=df_trabajo.index,  # Usar el índice como nombres de fondos
        x=df_trabajo['Overweight'],
        orientation='h',
        marker=dict(
            color=custom_colors[0],
            line=dict(color='white', width=1)  # Agregar borde blanco
        ),
        name='Outperformance'
    ))
    
    # Añadir una línea vertical en el eje x=0
    fig.add_shape(
        type="line",
        x0=0, x1=0, y0=-0.5, y1=len(df_trabajo)-0.5,
        line=dict(color="white", width=0.8)
    )
    
    # Añadir anotaciones para mostrar los valores
    annotations = []
    for i, (idx, row) in enumerate(df_trabajo.iterrows()):
        if row['Diferencia'] < 0:
            annotations.append(dict(
                x=row['Underweight'], 
                y=idx,  # Usar el índice como nombre del fondo
                text=f"{abs(row['Underweight']):.2f}", 
                xanchor='right', 
                showarrow=False, 
                font=dict(color='white')
            ))
        else:
            annotations.append(dict(
                x=row['Overweight'], 
                y=idx,  # Usar el índice como nombre del fondo
                text=f"{row['Overweight']:.2f}", 
                xanchor='left', 
                showarrow=False, 
                font=dict(color='white')
            ))
    
    # Ajustar altura del gráfico según la cantidad de fondos
    height = max(500, 100 + len(df_trabajo) * 40)  # 40px por fondo + margen
    
    # Configurar etiquetas y título
    fig.update_layout(
        annotations=annotations,
        xaxis_title='Diferencia de Rendimiento %',
        title=f'Diferencias de Rendimiento entre Fondo y Benchmark ({periodo}){" - " + titulo_adicional if titulo_adicional else ""}',
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        width=1000,
        margin=dict(l=200),
        plot_bgcolor='#0C2653',  # Color de fondo del gráfico
        paper_bgcolor='#0C2653',  # Color de fondo del papel
        font=dict(color='#FFFFFF')  # Color del texto
    )
    
    # Asegurar que todas las etiquetas se muestren completamente
    fig.update_yaxes(tickfont=dict(size=14))
    fig.update_xaxes(gridcolor='#44475A')
    fig.update_yaxes(gridcolor='#44475A')
    
    return fig

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
            "ACWI"
        ],
        "Pesos":[1.0]
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
            "SPESG", "IPC"
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

# Fecha = '2025-03-31'
# gf.verificador_cache_PreciosFondos(Fecha)

# df_p_r = pd.read_csv(f'../Data/Precios_Fondos{Fecha}.txt')

# st.dataframe(df_p_r)

bmks_rv = ["^MXX", "^SPESG", "^SPGSCI", "^GSPC", "^NDX"]

propiedades_fondos_path = "./ArchivosRendimientos/PropiedadesFondos"
propiedades_fondos_filename = os.listdir(propiedades_fondos_path)[0]
propiedades_fondos_df = pd.read_excel(os.path.join(propiedades_fondos_path, propiedades_fondos_filename))

precios_fondos_valmer_path = "./ArchivosPeergroups/PreciosFondosValmer"
precios_fondos_valmer_filename = os.listdir(precios_fondos_valmer_path)[0]
precios_fondos_valmer_df = pd.read_csv(os.path.join(precios_fondos_valmer_path, precios_fondos_valmer_filename))
precios_fondos_valmer_df = precios_fondos_valmer_df[["FECHA", "EMISORA", "SERIE", "PRECIO SUCIO"]]
precios_fondos_valmer_df.rename(columns={"FECHA":"Fecha", "EMISORA":"Fondo", "SERIE":"Serie", "PRECIO SUCIO":"Precio"}, inplace=True)
precios_fondos_valmer_df["Fecha"] = pd.to_datetime(precios_fondos_valmer_df["Fecha"], format="%Y-%m-%d")

precios_fondos_df = precios_fondos_valmer_df.copy()
precios_fondos_df = precios_fondos_df[
    pd.Series(list(zip(precios_fondos_df["Fondo"], precios_fondos_df["Serie"]))).isin(list(zip(propiedades_fondos_df["Fondo"], propiedades_fondos_df["Serie"])))
].reset_index(drop=True)

fechas = [datetime.today()]
temp_precios_fondos_df = descarga_rendimientos_MiVector(fechas)
temp_precios_fondos_df = temp_precios_fondos_df[
    pd.Series(list(zip(temp_precios_fondos_df["Fondo"], temp_precios_fondos_df["Serie"]))).isin(list(zip(propiedades_fondos_df["Fondo"], propiedades_fondos_df["Serie"])))
].reset_index(drop=True)
temp_precios_fondos_df = temp_precios_fondos_df[["Fecha", "Fondo", "Serie", "Precio"]]

precios_fondos_df = pd.concat([precios_fondos_df, temp_precios_fondos_df], axis=0, ignore_index=True)
precios_fondos_df = precios_fondos_df[["Fecha", "Fondo", "Precio"]].pivot(index="Fecha", columns="Fondo")
precios_fondos_df = precios_fondos_df.droplevel(level=0, axis=1)
precios_fondos_df.columns.name = None

fecha = datetime.today()
fecha = datetime(year=fecha.year, month=fecha.month, day=fecha.day)

xmex = xcals.get_calendar("XMEX")
fechas_bmv = sorted(xmex.sessions_in_range(
    start=(datetime.today() - relativedelta(years=20) + timedelta(days=100)).strftime("%Y-%m-%d"), 
    end=datetime.today().strftime("%Y-%m-%d")
).to_pydatetime())
fechas_bmv.remove(datetime(2024, 10, 1, 0, 0))

bmv_offset = infer_calendar(fechas_bmv)

fechas_exactas_iniciales_rf = calcula_fechas_exactas_iniciales(fecha)
fechas_habiles_iniciales_rf = calcula_fechas_habiles_iniciales(fechas_exactas_iniciales_rf, fechas_bmv, bmv_offset, tipo="Deuda")

fechas_exactas_iniciales_rv = calcula_fechas_exactas_iniciales((fecha - bmv_offset).to_pydatetime())
fechas_exactas_iniciales_rv = {
    ventana:(fechas_exactas_iniciales_rv[ventana] + bmv_offset).to_pydatetime() if ventana in ["MTD", "YTD"] else
    fechas_exactas_iniciales_rv[ventana] for ventana in fechas_exactas_iniciales_rv.keys()
}
fechas_habiles_iniciales_rv = calcula_fechas_habiles_iniciales(fechas_exactas_iniciales_rv, fechas_bmv, bmv_offset, tipo="RV")

precios_bmks_df = formatea_precios_bmks(fecha)

st.write(precios_fondos_df)
# st.write(precios_bmks_df)
# st.write(fechas_habiles_iniciales_rf)
# st.write(fechas_habiles_iniciales_rv)
# st.write(propiedades_fondos_df)

rendimientos_bmks_df = formatea_rendimientos_bmk(fecha, precios_bmks_df, fechas_habiles_iniciales_rf, fechas_habiles_iniciales_rv, propiedades_fondos_df, bmv_offset, fondo2benchmark)
rendimientos_bmks_df.reset_index(inplace=True)
rendimientos_bmks_df.rename(columns={"index":"Fondo"}|{col:f"BMK_{col}" for col in rendimientos_bmks_df.columns if col != "index"}, inplace=True)

rendimientos_fondos_df = formatea_rendimientos_fondos(fecha, precios_fondos_df, fechas_habiles_iniciales_rf, fechas_habiles_iniciales_rv, propiedades_fondos_df)
rendimientos_fondos_df.reset_index(inplace=True)
rendimientos_fondos_df.rename(columns={"index":"Fondo"}, inplace=True)

rendimientos_df = pd.merge(rendimientos_fondos_df, rendimientos_bmks_df, on="Fondo")
rendimientos_df = rendimientos_df[["Fondo"] + sum([[col, f"BMK_{col}"] for col in fechas_exactas_iniciales_rf.keys()], [])]
rendimientos_df.set_index("Fondo", inplace=True)
rendimientos_df *= 100
rendimientos_df = rendimientos_df.round(decimals=2)

st.write(rendimientos_df.style.format("{:.2f}"))

st.subheader("Rendimientos Históricos VS Benchmark")

fondo = st.selectbox(
    "Seleccione un fondo de Vector",
    tuple(rendimientos_fondos_df["Fondo"].unique())
)

st.write(precios_bmks_df[fondo2benchmark[fondo]["Benchmarks"]])
st.write(precios_fondos_df[fondo])

precios_fondo_bmks_df = pd.merge(precios_fondos_df[fondo].reset_index(), precios_bmks_df[fondo2benchmark[fondo]["Benchmarks"]].reset_index(), on="Fecha")
precios_fondo_bmks_df[fondo2benchmark[fondo]["Benchmarks"]] = precios_fondo_bmks_df[fondo2benchmark[fondo]["Benchmarks"]].shift(1)
precios_fondo_bmks_df.set_index("Fecha", inplace=True)
# rendimientos_fondo_bmks_df = precios_fondo_bmks_df.reset_index()
# rendimientos_fondo_bmks_df[[fondo] + fondo2benchmark[fondo]["Benchmarks"]] = rendimientos_fondo_bmks_df[[fondo] + fondo2benchmark[fondo]["Benchmarks"]].pct_change()
rendimientos_fondo_bmks_df = precios_fondo_bmks_df.pct_change()
rendimientos_fondo_bmks_df["BMK"] = (rendimientos_fondo_bmks_df[fondo2benchmark[fondo]["Benchmarks"]] * fondo2benchmark[fondo]["Pesos"]).sum(axis=1)
rendimientos_fondo_bmks_df.dropna(inplace=True)
rendimientos_fondo_bmks_df = rendimientos_fondo_bmks_df[[fondo, "BMK"]]

fecha_inicial_grafica_rendimientos_historicos = st.date_input(
    "Seleccione una fecha inicial:",
    value=rendimientos_fondo_bmks_df.index.min(),
    min_value=rendimientos_fondo_bmks_df.index.min(),
    max_value=rendimientos_fondo_bmks_df.index.max()
)

fecha_final_grafica_rendimientos_historicos = st.date_input(
    "Seleccione una fecha inicial:",
    value=rendimientos_fondo_bmks_df.index.max(),
    min_value=rendimientos_fondo_bmks_df.index.min(),
    max_value=rendimientos_fondo_bmks_df.index.max()
)

fecha_inicial_grafica_rendimientos_historicos = datetime(
    year=fecha_inicial_grafica_rendimientos_historicos.year,
    month=fecha_inicial_grafica_rendimientos_historicos.month,
    day=fecha_inicial_grafica_rendimientos_historicos.day
)
fecha_final_grafica_rendimientos_historicos = datetime(
    year=fecha_final_grafica_rendimientos_historicos.year,
    month=fecha_final_grafica_rendimientos_historicos.month,
    day=fecha_final_grafica_rendimientos_historicos.day
)

rendimientos_fondo_bmks_df = rendimientos_fondo_bmks_df.loc[
    (rendimientos_fondo_bmks_df.index >= fecha_inicial_grafica_rendimientos_historicos) &
    (rendimientos_fondo_bmks_df.index <= fecha_final_grafica_rendimientos_historicos)
]
st.write(rendimientos_fondo_bmks_df)

