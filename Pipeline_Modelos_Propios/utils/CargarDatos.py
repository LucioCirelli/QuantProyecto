# %%
import pandas as pd
import requests
import yfinance as yf
import numpy as np
import warnings
import random
from scipy import stats
from scipy.stats import (norm, t, shapiro, jarque_bera, kstest,
                         lognorm, gamma, expon, weibull_min, beta,
                         gumbel_r, logistic, laplace, cauchy, chi2,
                         exponweib, genextreme, genpareto, levy_stable)


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


def proyectar_rendimientos(df, alpha=0.94, nan_threshold=0.1):
    """
    Calcula rendimientos esperados y matriz de covarianzas (EWMA) usando precios de cierre.
    Mantiene la mayoría de los tickers, rellenando los pocos NaN.

    Parámetros:
        df : DataFrame con columnas ['Date', 'Ticker', 'Close']
        alpha : decaimiento exponencial (0 < alpha < 1), típico 0.94
        nan_threshold : fracción máxima de NaN permitida por ticker (ej: 0.1 = 10%)

    Retorna:
        mu_pred : Series con rendimientos esperados por ticker
        cov_ewma : DataFrame con matriz de covarianzas suavizada
    """

    # --- Validación ---
    if not {'Date', 'Ticker', 'Close'}.issubset(df.columns):
        raise ValueError("El DataFrame debe contener columnas ['Date', 'Ticker', 'Close']")

    # --- Pivot a formato ancho ---
    precios = df.pivot(index='Date', columns='Ticker', values='Close').sort_index()

    # --- Eliminar tickers con demasiados NaN ---
    precios = precios.loc[:, precios.isna().mean() <= nan_threshold]

    # --- Rellenar NaN restantes con forward-fill y back-fill ---
    precios = precios.ffill().bfill()

    # --- Calcular rendimientos logarítmicos ---
    rend = np.log(precios / precios.shift(1)).dropna(how='all')

    # --- Suavización exponencial para rendimientos esperados ---
    mu_ewma_series = rend.ewm(alpha=1-alpha, adjust=False).mean()
    mu_pred = mu_ewma_series.iloc[-1]

    # --- Matriz de covarianzas EWMA (centrada) ---
    cov_ewma = np.zeros((len(rend.columns), len(rend.columns)))
    for i in range(len(rend)):
        r = rend.iloc[i].values.reshape(-1, 1)
        m = mu_ewma_series.iloc[i].values.reshape(-1, 1)
        r_centered = r - m
        cov_ewma = alpha * cov_ewma + (1 - alpha) * (r_centered @ r_centered.T)
    cov_ewma = pd.DataFrame(cov_ewma, index=rend.columns, columns=rend.columns)

    return mu_pred, cov_ewma


def calcular_portfolio_stats(mu_pred, cov_ewma, pesos):
    """
    Calcula rendimiento esperado y volatilidad de un portafolio.

    pesos: Series con índice = tickers seleccionados.
    """
    tickers = pesos.index

    # Subseleccionar mu_pred y cov_ewma según tickers del portafolio
    mu_sub = mu_pred.loc[tickers]
    cov_sub = cov_ewma.loc[tickers, tickers]

    # Convertir pesos a array y normalizar
    w = np.array(pesos)
    w = w / w.sum()

    # Rendimiento esperado
    r_port = np.dot(w, mu_sub.values)

    # Volatilidad
    vol_port = np.sqrt(w.T @ cov_sub.values @ w)

    return {'Rendimiento': r_port, 'Volatilidad': vol_port}


def ajustar_distribucion_accion(df, ticker, verbose=True, umbral_ajuste=0.05, nivel_confianza=0.95):
    """
    Ajusta los rendimientos de una acción a múltiples distribuciones de probabilidad continuas
    y selecciona la mejor. Calcula la probabilidad de rendimiento negativo, VaR y CVaR.
    Si ninguna distribución ajusta bien, usa método no paramétrico (empírico).
    
    Parámetros:
        df : DataFrame con columnas ['Date', 'Ticker', 'Close']
        ticker : str, símbolo de la acción a analizar
        verbose : bool, si True imprime resultados detallados
        umbral_ajuste : float, p-value mínimo del test KS para considerar un buen ajuste (default 0.05)
        nivel_confianza : float, nivel de confianza para VaR y CVaR (default 0.95)
    
    Retorna:
        dict con:
            - 'rendimientos': Series con rendimientos logarítmicos
            - 'estadisticas': dict con media, desviación, skewness, kurtosis
            - 'mejor_distribucion': nombre de la mejor distribución (o 'Empírico')
            - 'parametros_mejor': parámetros de la mejor distribución (o None si es empírico)
            - 'prob_rendimiento_negativo': probabilidad P(R < 0)
            - 'var': Value at Risk al nivel de confianza especificado
            - 'cvar': Conditional Value at Risk (Expected Shortfall)
            - 'ajuste_aceptable': bool, True si el ajuste es estadísticamente aceptable
            - 'metodo_usado': 'parametrico' o 'no_parametrico'
            - 'prob_empirica': probabilidad calculada por ocurrencias históricas
            - 'ic_empirico': intervalo de confianza 95% de la probabilidad empírica
            - 'resultados_todas': lista con resultados de todas las distribuciones probadas
            - 'tests_normalidad': dict con resultados de tests estadísticos
    """
    
    # Filtrar datos del ticker
    df_ticker = df[df['Ticker'] == ticker].copy()
    
    if len(df_ticker) == 0:
        raise ValueError(f"No se encontraron datos para el ticker {ticker}")
    
    # Ordenar por fecha y preparar precios
    df_ticker = df_ticker.sort_values('Date')
    precios = df_ticker['Close'].values
    
    # Calcular rendimientos logarítmicos
    rendimientos = np.log(precios[1:] / precios[:-1])
    rendimientos = pd.Series(rendimientos, index=df_ticker['Date'].iloc[1:])
    
    # Eliminar posibles NaN o inf
    rendimientos = rendimientos.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(rendimientos) < 30:
        raise ValueError(f"Insuficientes datos para {ticker} (mínimo 30 observaciones)")
    
    # --- CÁLCULO EMPÍRICO (NO PARAMÉTRICO) ---
    # Probabilidad empírica basada en ocurrencias históricas
    n_negativos = (rendimientos < 0).sum()
    n_total = len(rendimientos)
    prob_empirica = n_negativos / n_total
    
    # Intervalo de confianza usando distribución binomial (95%)
    # Aproximación normal: p ± 1.96 * sqrt(p*(1-p)/n)
    se_empirica = np.sqrt(prob_empirica * (1 - prob_empirica) / n_total)
    ic_lower = max(0, prob_empirica - 1.96 * se_empirica)
    ic_upper = min(1, prob_empirica + 1.96 * se_empirica)
    
    # --- Estadísticas descriptivas ---
    media = rendimientos.mean()
    desv = rendimientos.std()
    skew = stats.skew(rendimientos)
    kurt = stats.kurtosis(rendimientos)
    
    estadisticas = {
        'media': media,
        'desviacion': desv,
        'skewness': skew,
        'kurtosis': kurt,
        'n_observaciones': len(rendimientos),
        'minimo': rendimientos.min(),
        'maximo': rendimientos.max(),
        'n_negativos': n_negativos,
        'n_positivos': n_total - n_negativos,
        'n_ceros': (rendimientos == 0).sum()
    }
    
    # --- Definir distribuciones a probar ---
    distribuciones = {
        'Normal': norm,
        't-Student': t,
        'Laplace': laplace,
        'Logistic': logistic,
        'Cauchy': cauchy,
        'Gumbel': gumbel_r,
        'Generalized Extreme Value': genextreme,
        'Generalized Pareto': genpareto,
        'Exponential Weibull': exponweib,
    }
    
    # Solo para datos positivos
    if rendimientos.min() > 0:
        distribuciones.update({
            'Lognormal': lognorm,
            'Gamma': gamma,
            'Weibull': weibull_min,
            'Exponential': expon,
            'Chi-squared': chi2,
        })
    
    # --- Ajustar todas las distribuciones ---
    resultados = []
    
    for nombre, dist in distribuciones.items():
        try:
            # Ajustar parámetros
            params = dist.fit(rendimientos)
            
            # Test de Kolmogorov-Smirnov
            ks_stat, ks_pvalue = kstest(rendimientos, nombre.lower().replace(' ', '').replace('-', ''), 
                                        args=params)
            
            # Calcular probabilidad de rendimiento negativo
            prob_neg = dist.cdf(0, *params)
            
            # AIC y BIC para comparación
            log_likelihood = np.sum(dist.logpdf(rendimientos, *params))
            k = len(params)  # número de parámetros
            n = len(rendimientos)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            resultados.append({
                'distribucion': nombre,
                'parametros': params,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'prob_rendimiento_negativo': prob_neg,
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'n_parametros': k
            })
            
        except Exception as e:
            if verbose:
                print(f"   ⚠️ No se pudo ajustar {nombre}: {str(e)[:50]}")
            continue
    
    # --- Ordenar por p-value de KS (mayor es mejor) y luego por AIC (menor es mejor) ---
    if len(resultados) == 0:
        raise ValueError(f"No se pudo ajustar ninguna distribución para {ticker}")
    
    resultados_df = pd.DataFrame(resultados)
    resultados_df = resultados_df.sort_values(['ks_pvalue', 'aic'], ascending=[False, True])
    
    mejor = resultados_df.iloc[0]
    mejor_dist = mejor['distribucion']
    mejor_params = mejor['parametros']
    mejor_ks_pvalue = mejor['ks_pvalue']
    prob_neg_parametrica = mejor['prob_rendimiento_negativo']
    
    ajuste_aceptable = mejor_ks_pvalue > umbral_ajuste
    
    # --- Decidir si usar método paramétrico o no paramétrico ---
    if ajuste_aceptable:
        metodo_usado = 'parametrico'
        prob_final = prob_neg_parametrica
        distribucion_final = mejor_dist
        parametros_final = mejor_params
        
        # VaR y CVaR paramétricos usando la mejor distribución
        alpha = 1 - nivel_confianza  # ej: 0.05 para 95% confianza
        dist_obj = distribuciones[mejor_dist]
        
        # VaR: cuantil al nivel alpha (pérdida máxima esperada)
        var = dist_obj.ppf(alpha, *mejor_params)
        
        # CVaR (Expected Shortfall): esperanza condicional de pérdidas más allá del VaR
        # CVaR = E[R | R <= VaR] = (1/alpha) * integral de r*f(r) de -inf a VaR
        # Para distribuciones continuas: CVaR ≈ media de la cola izquierda
        # Aproximación mediante simulación
        n_sim = 100000
        simulaciones = dist_obj.rvs(*mejor_params, size=n_sim)
        perdidas_extremas = simulaciones[simulaciones <= var]
        cvar = perdidas_extremas.mean() if len(perdidas_extremas) > 0 else var
        
    else:
        metodo_usado = 'no_parametrico'
        prob_final = prob_empirica
        distribucion_final = 'Empírico (Frecuencias Históricas)'
        parametros_final = None
        
        # VaR y CVaR empíricos usando cuantiles históricos
        alpha = 1 - nivel_confianza
        var = rendimientos.quantile(alpha)
        
        # CVaR empírico: promedio de rendimientos peores que VaR
        perdidas_extremas = rendimientos[rendimientos <= var]
        cvar = perdidas_extremas.mean() if len(perdidas_extremas) > 0 else var
    
    # --- Tests de normalidad (para referencia) ---
    # Shapiro-Wilk (solo si n < 5000)
    if len(rendimientos) <= 5000:
        shapiro_stat, shapiro_p = shapiro(rendimientos)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
        
    # Jarque-Bera
    jb_stat, jb_p = jarque_bera(rendimientos)
    
    tests_normalidad = {
        'shapiro': {'estadistico': shapiro_stat, 'p_value': shapiro_p},
        'jarque_bera': {'estadistico': jb_stat, 'p_value': jb_p},
    }
    
    # --- Imprimir resultados si verbose=True ---
    if verbose:
        print(f"\n{'='*80}")
        print(f"ANÁLISIS DE DISTRIBUCIÓN PARA {ticker}")
        print(f"{'='*80}")
        print(f"\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
        print(f"   • N observaciones: {estadisticas['n_observaciones']}")
        print(f"   • Media:           {estadisticas['media']:.6f}")
        print(f"   • Desv. Estándar:  {estadisticas['desviacion']:.6f}")
        print(f"   • Skewness:        {estadisticas['skewness']:.6f}")
        print(f"   • Kurtosis:        {estadisticas['kurtosis']:.6f}")
        print(f"   • Rango:           [{estadisticas['minimo']:.6f}, {estadisticas['maximo']:.6f}]")
        print(f"   • Rendim. neg.:    {n_negativos} ({prob_empirica*100:.2f}%)")
        print(f"   • Rendim. pos.:    {estadisticas['n_positivos']} ({(1-prob_empirica)*100:.2f}%)")
        
        print(f"\n� TESTS DE NORMALIDAD:")
        if not np.isnan(shapiro_p):
            resultado_sw = "Rechaza normalidad" if shapiro_p < 0.05 else "No rechaza normalidad"
            print(f"   • Shapiro-Wilk:    stat={shapiro_stat:.4f}, p={shapiro_p:.4f} → {resultado_sw}")
        resultado_jb = "Rechaza normalidad" if jb_p < 0.05 else "No rechaza normalidad"
        print(f"   • Jarque-Bera:     stat={jb_stat:.4f}, p={jb_p:.4f} → {resultado_jb}")
        
        print(f"\n� TOP 5 MEJORES AJUSTES (ordenado por p-value KS):")
        print(f"{'Rank':<6} {'Distribución':<25} {'KS p-value':<12} {'AIC':<12} {'P(R<0)':<10}")
        print(f"{'-'*80}")
        for idx, row in resultados_df.head(5).iterrows():
            ajuste_ok = "✓" if row['ks_pvalue'] > umbral_ajuste else "✗"
            print(f"{ajuste_ok:<6} {row['distribucion']:<25} {row['ks_pvalue']:<12.4f} "
                  f"{row['aic']:<12.2f} {row['prob_rendimiento_negativo']:<10.4f}")
        
        print(f"\n{'='*80}")
        
        if ajuste_aceptable:
            print(f"✅ MÉTODO: PARAMÉTRICO - {mejor_dist}")
            print(f"{'='*80}")
            print(f"   • Parámetros:      {mejor_params}")
            print(f"   • KS Statistic:    {mejor['ks_statistic']:.6f}")
            print(f"   • KS p-value:      {mejor_ks_pvalue:.6f}")
            print(f"   • AIC:             {mejor['aic']:.2f}")
            print(f"   • BIC:             {mejor['bic']:.2f}")
            print(f"   • P(R < 0):        {prob_final:.4f} ({prob_final*100:.2f}%)")
            print(f"\n   📉 MEDIDAS DE RIESGO ({nivel_confianza*100:.0f}% confianza):")
            print(f"   • VaR:             {var:.6f} ({var*100:.2f}%)")
            print(f"   • CVaR (ES):       {cvar:.6f} ({cvar*100:.2f}%)")
            print(f"\n   ✅ AJUSTE ACEPTABLE (p-value > {umbral_ajuste})")
        else:
            print(f"⚠️ MÉTODO: NO PARAMÉTRICO (EMPÍRICO)")
            print(f"{'='*80}")
            print(f"   Los datos NO se ajustan bien a ninguna distribución estándar.")
            print(f"   Se usará estimación empírica basada en frecuencias históricas.")
            print(f"\n   📊 PROBABILIDAD EMPÍRICA:")
            print(f"   • P(R < 0):        {prob_empirica:.4f} ({prob_empirica*100:.2f}%)")
            print(f"   • IC 95%:          [{ic_lower:.4f}, {ic_upper:.4f}]")
            print(f"   • Basado en:       {n_negativos}/{n_total} observaciones negativas")
            print(f"\n   � MEDIDAS DE RIESGO ({nivel_confianza*100:.0f}% confianza - Empíricas):")
            print(f"   • VaR:             {var:.6f} ({var*100:.2f}%)")
            print(f"   • CVaR (ES):       {cvar:.6f} ({cvar*100:.2f}%)")
            print(f"\n   �💡 RECOMENDACIÓN:")
            print(f"      - Use métodos no paramétricos (bootstrap, simulación histórica)")
            print(f"      - Considere modelos GARCH para volatilidad variable")
            print(f"      - Evalúe transformaciones de datos (Box-Cox, log)")
        
        print(f"\n   📌 COMPARACIÓN:")
        print(f"   • Mejor paramétrica:  {prob_neg_parametrica:.4f} ({mejor_dist})")
        print(f"   • Empírica:           {prob_empirica:.4f} (IC: [{ic_lower:.4f}, {ic_upper:.4f}])")
        
        print(f"{'='*80}\n")
    
    return {
        'rendimientos': rendimientos,
        'estadisticas': estadisticas,
        'mejor_distribucion': distribucion_final,
        'parametros_mejor': parametros_final,
        'prob_rendimiento_negativo': prob_final,
        'var': var,
        'cvar': cvar,
        'nivel_confianza': nivel_confianza,
        'prob_empirica': prob_empirica,
        'ic_empirico': (ic_lower, ic_upper),
        'ajuste_aceptable': ajuste_aceptable,
        'metodo_usado': metodo_usado,
        'ks_statistic': mejor['ks_statistic'] if ajuste_aceptable else None,
        'ks_pvalue': mejor_ks_pvalue,
        'aic': mejor['aic'] if ajuste_aceptable else None,
        'bic': mejor['bic'] if ajuste_aceptable else None,
        'resultados_todas': resultados_df,
        'tests_normalidad': tests_normalidad
    }


# ============================================================================
# CÓDIGO DE EJEMPLO (NO SE EJECUTA AL IMPORTAR)
# ============================================================================

if __name__ == "__main__":
    df_datos = descargar_sp500_mensual(start_year=2000, end_year=2015)
    
    mu_pred, cov_ewma = proyectar_rendimientos(df_datos, alpha=0.94)
    
    # Supongamos que 'precios' es tu DataFrame con columnas: ['Date', 'Ticker', 'Close']
    # Calculamos rendimientos logarítmicos históricos
    rend_hist = df_datos.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    rend_hist = np.log(rend_hist / rend_hist.shift(1))
    rend_hist = rend_hist.loc[:, rend_hist.isna().mean() < 0.1]
    rend_hist = rend_hist.ffill().bfill()
    
    # Número de portafolios a generar
    n_portfolios = 1000
    n_seed_tickers = 50  # tickers que alimentan el modelo
    n_selected_tickers = 10  # tickers que el modelo selecciona
    
    portfolios_weights = []  # lista para guardar pesos de cada portafolio
    portfolios_returns = []  # lista para guardar serie temporal de cada portafolio
    
    tickers_disponibles = rend_hist.columns.tolist()
    
    for i in range(n_portfolios):
        #Seleccionar 50 tickers aleatorios para alimentar el modelo
        tickers_seed = random.sample(tickers_disponibles, n_seed_tickers)
        
        #Simular modelo que elige 10 tickers de esos 50
        tickers_selected = random.sample(tickers_seed, n_selected_tickers)
        
        #Asignar pesos aleatorios que sumen 1
        pesos = np.random.random(n_selected_tickers)
        pesos /= pesos.sum()
        
        pesos_series = pd.Series(pesos, index=tickers_selected)
        portfolios_weights.append(pesos_series)
        
        #Calcular serie temporal de rendimientos ponderados del portafolio
        rend_port = rend_hist[tickers_selected].dot(pesos_series.values)
        portfolios_returns.append(rend_port)
    
    # Convertir a DataFrame: filas = fechas, columnas = portafolios
    df_portfolios = pd.concat(portfolios_returns, axis=1)
    df_portfolios.columns = [f'Portfolio_{i}' for i in range(n_portfolios)]
    
    # Calcular estadísticas de cada portafolio
    df_stats = pd.DataFrame({
        'Rendimiento': df_portfolios.mean(),
        'Volatilidad': df_portfolios.std()
    })
    
    #Calcular matriz de covarianzas y correlaciones entre portafolios
    cov_portfolios = df_portfolios.cov()
    corr_portfolios = df_portfolios.corr()
    
    
    def ewma_portfolios(df_portfolios, alpha=0.94):
        """
        Calcula rendimiento esperado anual y matriz de covarianzas EWMA para portafolios.

        Parámetros:
            df_portfolios : DataFrame con columnas = portafolios, filas = rendimientos históricos
            alpha : factor de suavización exponencial (0 < alpha < 1)

        Retorna:
            mu_port : Series con rendimiento anual esperado de cada portafolio
            cov_port : DataFrame con matriz de covarianzas anualizada entre portafolios
        """
        n_periods = 12  # asumimos datos mensuales para anualizar

        # --- Rendimiento esperado EWMA ---
        mu_ewma = df_portfolios.ewm(alpha=1 - alpha, adjust=False).mean().iloc[-1]

        # Anualizamos el rendimiento
        mu_port = mu_ewma * n_periods

        # --- Matriz de covarianzas EWMA ---
        tickers = df_portfolios.columns
        cov_ewma = np.zeros((len(tickers), len(tickers)))

        for i in range(len(df_portfolios)):
            r = df_portfolios.iloc[i].values.reshape(-1, 1)
            cov_ewma = alpha * cov_ewma + (1 - alpha) * (r @ r.T)

        # Convertir a DataFrame y anualizar (multiplicamos por n_periods para varianza anual)
        cov_port = pd.DataFrame(cov_ewma * n_periods, index=tickers, columns=tickers)

        return mu_port, cov_port
    
    
    mu_port, cov_port = ewma_portfolios(df_portfolios)
    
    
    # Comparacion
    
    precios_2016 = descargar_sp500_mensual(start_year=2016, end_year=2016)
    # 2️⃣ Obtener precios de 2016
    precios_2016 = precios_2016[precios_2016['Date'].dt.year == 2016]
    precios_2016 = precios_2016.pivot(index='Date', columns='Ticker', values='Close')
    precios_2016 = precios_2016.ffill().bfill()  # rellenar NaN
    
    resultados_2016 = []
    
    tickers = pesos.index
    portf_name = f'Portfolio_{i}'
    
    # Seleccionar precios de los tickers del portafolio
    precios_pf = precios_2016[tickers]
    
    # Calcular rendimientos logarítmicos
    rend_2016 = np.log(precios_pf / precios_pf.shift(1)).dropna(how='all')
    
    # Rendimiento real anual del portafolio
    rend_real = rend_2016.dot(pesos.values).sum()
    
    # Rendimiento predicho (EWMA)
    rend_pred = mu_port[portf_name]
    
    # Guardar resultado
    resultados_2016.append({
        'Portfolio': portf_name,
        'Rendimiento_Predicho': rend_pred,
        'Rendimiento_Real': rend_real,
        'Diferencia': rend_real - rend_pred
    })
    
    # Convertir a DataFrame
    df_comparacion = pd.DataFrame(resultados_2016)
    print(df_comparacion.head())
    
    
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

# ============================================================================
# EJEMPLO DE USO: Ajustar distribución para una acción específica
# ============================================================================
# La función ahora prueba múltiples distribuciones continuas y selecciona la mejor.
# Si ninguna ajusta bien (p-value KS ≤ umbral), usa método NO PARAMÉTRICO (empírico).
# 
# resultado = ajustar_distribucion_accion(df_datos, 'AAPL', verbose=True, umbral_ajuste=0.05)
# 
# # Acceder a resultados:
# # resultado['mejor_distribucion']  # Nombre de la mejor distribución (o 'Empírico')
# # resultado['metodo_usado']  # 'parametrico' o 'no_parametrico'
# # resultado['prob_rendimiento_negativo']  # P(R < 0) según mejor método
# # resultado['prob_empirica']  # P(R < 0) calculada por frecuencias históricas
# # resultado['ic_empirico']  # Intervalo de confianza 95% del método empírico
# # resultado['ajuste_aceptable']  # True/False si el ajuste es estadísticamente aceptable
# # resultado['parametros_mejor']  # Parámetros (o None si es empírico)
# # resultado['resultados_todas']  # DataFrame con resultados de todas las distribuciones
# 
# # Ver todas las distribuciones probadas y comparar:
# # print(resultado['resultados_todas'][['distribucion', 'ks_pvalue', 'aic', 'prob_rendimiento_negativo']])
