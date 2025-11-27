import requests
import yfinance as yf
import pandas as pd


def descargar_sp500_mensual(start_year=2015, end_year=2025, guardar_csv=False):
    """
    Descarga precios mensuales (Close) de los tickers del S&P 500 entre los años especificados.
    Devuelve un DataFrame plano: ['Date', 'Ticker', 'Close'].
    """
    # --- Obtener tickers del S&P500 desde Wikipedia ---
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tabla = pd.read_html(response.text)
    tickers = tabla[1]['Symbol'].to_list()

    # --- Rango temporal ---
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    # --- Descargar datos mensuales ---
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval='1mo',
        group_by='ticker',
        auto_adjust=True,
        progress=False,
        threads=True
    )

    registros = []
    # --- Si yfinance devuelve MultiIndex (típico) ---
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in data.columns.levels[0]:
            try:
                df_t = data[ticker][['Close']].reset_index()
                df_t['Ticker'] = ticker
                registros.append(df_t)
            except KeyError:
                continue
        df_final = pd.concat(registros, ignore_index=True)

    else:
        # --- Caso raro: columnas planas ---
        if 'Close' not in data.columns:
            raise ValueError("No se encontró columna 'Close' en los datos descargados.")
        df_final = data[['Close']].reset_index()
        df_final['Ticker'] = tickers[0]  # si solo descarga uno

    # --- Guardar si se desea ---
    if guardar_csv:
        nombre = f"sp500_close_mensual_{start_year}_{end_year}.csv"
        df_final.to_csv(nombre, index=False)
        print(f"Datos guardados en {nombre}")

    return df_final




def descargar_spy(start_year=2015, end_year=2025):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tabla = pd.read_html(response.text)
    tickers = tabla[1]['Symbol'].to_list()

    # --- Rango temporal ---
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    # --- Descargar datos mensuales ---
    data = yf.download(
        tickers=["SPY"],
        start=start_date,
        end=end_date,
        interval='1mo',
        group_by='ticker',
        auto_adjust=True,
        progress=False,
        threads=True
    )
    registros = []
    # --- Si yfinance devuelve MultiIndex (típico) ---
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in data.columns.levels[0]:
            try:
                df_t = data[ticker][['Close']].reset_index()
                df_t['Ticker'] = ticker
                registros.append(df_t)
            except KeyError:
                continue
        df_final = pd.concat(registros, ignore_index=True)

    else:
        # --- Caso raro: columnas planas ---
        if 'Close' not in data.columns:
            raise ValueError("No se encontró columna 'Close' en los datos descargados.")
        df_final = data[['Close']].reset_index()
        df_final['Ticker'] = tickers[0]  # si solo descarga uno
    return df_final


