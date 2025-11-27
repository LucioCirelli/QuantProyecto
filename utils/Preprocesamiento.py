"""
PREPROCESAMIENTO.PY - Generación de inputs para el modelo estocástico

Genera diccionario inputs_modelo con:
- set_acciones: lista de tickers
- parámetros por acción: mu, sigma, prob_neg, var, cvar
- matriz de covarianzas
- guarda pickle para ser leído por el modelo

Uso:
    from Preprocesamiento import generar_inputs_modelo
    inputs = generar_inputs_modelo(df_datos)
"""

import pandas as pd
import numpy as np
import pickle
import os
from utils.CargarDatos import proyectar_rendimientos, ajustar_distribucion_accion


def generar_inputs_modelo(df_datos, alpha_ewma=0.94, max_activos=None, 
                          guardar_pickle=True, nombre_archivo='inputs_modelo.pkl'):
    """
    Genera diccionario de inputs para el modelo estocástico.
    
    Parámetros:
        df_datos : DataFrame con columnas ['Date', 'Ticker', 'Close'] y opcionalmente 'Return'
        alpha_ewma : float, parámetro EWMA (default 0.94) - DEPRECADO, ahora usa método simple
        max_activos : int, límite de activos a procesar (None = todos)
        guardar_pickle : bool, si True guarda el diccionario en pickle
        nombre_archivo : str, nombre del archivo pickle
    
    Retorna:
        dict con estructura compatible con ModeloEstocastico.py y Pipeline_Franco:
            - set_acciones: list de tickers
            - rendimiento_esperado: dict {ticker: mu}
            - desvio_estandar: dict {ticker: sigma}
            - covarianzas: dict {(ticker_i, ticker_j): cov}
            - probabilidad_perdida: dict {ticker: P(R<0)}
            - var: dict {ticker: VaR al 95%}
            - cvar: dict {ticker: CVaR al 95%}
            - metadata: dict con información de la corrida
    
    Nota: Ahora utiliza el método de Pipeline_Franco (media simple + momentum)
          en lugar de EWMA para compatibilidad con OptimizarCartera.py
    """
    
    print("\n📊 Generando inputs para modelo estocástico...")
    
    # 1. NUEVO: Calcular rendimientos esperados como en DinamicMVO de Franco
    print("   Calculando rendimientos esperados (método Franco)...")
    
    # Verificar si ya tiene columna Return (como en descargar_tickets.py)
    if 'Return' not in df_datos.columns:
        # Si no tiene, calcular rendimientos logarítmicos
        df_datos = df_datos.copy()
        df_datos['Date'] = pd.to_datetime(df_datos['Date'])
        df_datos = df_datos.sort_values(['Ticker', 'Date'])
        df_datos['Return'] = df_datos.groupby('Ticker')['Close'].transform(lambda x: np.log(x / x.shift(1)))
        df_datos = df_datos.dropna(subset=['Return'])
    
    # Calcular μ = 0.5*media + 0.5*momentum (últimos 3 meses)
    mean_return = df_datos.groupby('Ticker')['Return'].mean()
    momentum = df_datos.groupby('Ticker').apply(lambda x: x.sort_values('Date').tail(3)['Return'].mean())
    mu_pred = 0.5 * mean_return + 0.5 * momentum
    
    # Rellenar NaN con mediana si existen
    if mu_pred.isna().any():
        mu_pred = mu_pred.fillna(mu_pred.median())
    
    # Calcular matriz de covarianzas (método simple, no EWMA)
    print("   Calculando matriz de covarianzas...")
    pivot_returns = df_datos.pivot(index='Date', columns='Ticker', values='Return')
    cov_matrix = pivot_returns.cov()
    
    # 2. Seleccionar activos (por mayor rendimiento esperado)
    mu_pred_sorted = mu_pred.sort_values(ascending=False)
    tickers = mu_pred_sorted.index.tolist()
    
    if max_activos:
        tickers = tickers[:max_activos]
        mu_pred = mu_pred.loc[tickers]
        cov_matrix = cov_matrix.loc[tickers, tickers]
    
    print(f"   Procesando {len(tickers)} activos...")
    
    # 3. Calcular volatilidades (desviaciones estándar)
    desvio_estandar = {ticker: np.sqrt(cov_matrix.loc[ticker, ticker]) for ticker in tickers}
    sigma_array = np.array([desvio_estandar[t] for t in tickers])
    
    # 4. Calcular delta (incertidumbre) como en Pipeline_Franco
    print("   Calculando intervalos de incertidumbre (delta)...")
    T = len(df_datos['Date'].unique())  # Número de períodos
    delta_array = 1.96 * sigma_array / np.sqrt(T)  # Intervalo de confianza 95%
    delta_dict = {ticker: delta_array[i] for i, ticker in enumerate(tickers)}
    
    # 5. Métricas de riesgo simplificadas (método empírico rápido)
    print("   Calculando métricas de riesgo...")
    probabilidad_perdida = {}
    var_dict = {}
    cvar_dict = {}
    
    for ticker in tickers:
        ticker_returns = df_datos[df_datos['Ticker'] == ticker]['Return'].dropna()
        
        # Probabilidad empírica de rendimiento negativo
        prob_neg = (ticker_returns < 0).sum() / len(ticker_returns) if len(ticker_returns) > 0 else 0.5
        probabilidad_perdida[ticker] = prob_neg
        
        # VaR y CVaR al 95% (percentiles empíricos)
        if len(ticker_returns) > 0:
            var_dict[ticker] = ticker_returns.quantile(0.05)  # 5th percentile
            # CVaR = media de retornos peores que VaR
            returns_below_var = ticker_returns[ticker_returns <= var_dict[ticker]]
            cvar_dict[ticker] = returns_below_var.mean() if len(returns_below_var) > 0 else var_dict[ticker]
        else:
            var_dict[ticker] = -2 * desvio_estandar[ticker]
            cvar_dict[ticker] = -3 * desvio_estandar[ticker]
    
    print(f"   ✅ Análisis completado (método empírico rápido)")
    
    # 6. Convertir matriz de covarianzas a diccionario
    covarianzas = {(i, j): cov_matrix.loc[i, j] 
                   for i in tickers for j in tickers}
    
    # 7. Construir diccionario de inputs (compatible con ModeloEstocastico.py Y Pipeline_Franco)
    inputs_modelo = {
        'set_acciones': tickers,
        'rendimiento_esperado': mu_pred.to_dict(),
        'desvio_estandar': desvio_estandar,
        'delta': delta_dict,  # NUEVO: Intervalo de incertidumbre para Robust Optimization
        'covarianzas': covarianzas,
        'probabilidad_perdida': probabilidad_perdida,
        'var': var_dict,
        'cvar': cvar_dict,
        'metadata': {
            'n_activos': len(tickers),
            'metodo': 'simple_momentum',  # Cambio de EWMA a método simple+momentum
            'T_periodos': T,
            'fecha_generacion': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    # 8. Guardar pickle si se solicita
    if guardar_pickle:
        with open(nombre_archivo, 'wb') as f:
            pickle.dump(inputs_modelo, f)
        print(f"   💾 Archivo guardado: {nombre_archivo}")
    
    print(f"   ✅ Inputs generados exitosamente")
    print(f"      • Tickers: {len(tickers)}")
    print(f"      • μ rango: [{mu_pred.min():.6f}, {mu_pred.max():.6f}]")
    print(f"      • σ rango: [{min(desvio_estandar.values()):.6f}, {max(desvio_estandar.values()):.6f}]")
    
    return inputs_modelo


def cargar_inputs_modelo(nombre_archivo='inputs_modelo.pkl'):
    """
    Carga diccionario de inputs desde archivo pickle.
    
    Parámetros:
        nombre_archivo : str, ruta al archivo pickle
    
    Retorna:
        dict con inputs del modelo
    """
    if not os.path.exists(nombre_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {nombre_archivo}")
    
    with open(nombre_archivo, 'rb') as f:
        inputs_modelo = pickle.load(f)
    
    print(f"✅ Inputs cargados desde: {nombre_archivo}")
    print(f"   Activos: {inputs_modelo['metadata']['n_activos']}")
    print(f"   Fecha: {inputs_modelo['metadata']['fecha_generacion']}")
    
    return inputs_modelo


def resumen_inputs(inputs_modelo):
    """
    Imprime resumen de los inputs generados.
    
    Parámetros:
        inputs_modelo : dict generado por generar_inputs_modelo()
    """
    print("\n" + "="*80)
    print("RESUMEN DE INPUTS PARA MODELO ESTOCÁSTICO")
    print("="*80)
    
    meta = inputs_modelo['metadata']
    print(f"\n📊 Información General:")
    print(f"   • Número de activos:  {meta['n_activos']}")
    print(f"   • Método:             {meta['metodo']}")
    print(f"   • Períodos (T):       {meta['T_periodos']}")
    print(f"   • Fecha generación:   {meta['fecha_generacion']}")
    
    # Estadísticas de rendimientos
    mu_vals = list(inputs_modelo['rendimiento_esperado'].values())
    print(f"\n📈 Rendimientos Esperados (μ):")
    print(f"   • Media:              {np.mean(mu_vals):.6f}")
    print(f"   • Mediana:            {np.median(mu_vals):.6f}")
    print(f"   • Rango:              [{np.min(mu_vals):.6f}, {np.max(mu_vals):.6f}]")
    
    # Estadísticas de volatilidad
    sigma_vals = list(inputs_modelo['desvio_estandar'].values())
    print(f"\n📊 Volatilidades (σ):")
    print(f"   • Media:              {np.mean(sigma_vals):.6f}")
    print(f"   • Mediana:            {np.median(sigma_vals):.6f}")
    print(f"   • Rango:              [{np.min(sigma_vals):.6f}, {np.max(sigma_vals):.6f}]")
    
    # Estadísticas de probabilidad negativa
    prob_vals = list(inputs_modelo['probabilidad_perdida'].values())
    print(f"\n⚠️ Probabilidad Rendimiento Negativo P(R<0):")
    print(f"   • Media:              {np.mean(prob_vals):.4f} ({np.mean(prob_vals)*100:.2f}%)")
    print(f"   • Mediana:            {np.median(prob_vals):.4f} ({np.median(prob_vals)*100:.2f}%)")
    print(f"   • Rango:              [{np.min(prob_vals):.4f}, {np.max(prob_vals):.4f}]")
    
    # Estadísticas de VaR
    var_vals = list(inputs_modelo['var'].values())
    print(f"\n📉 Value at Risk (VaR 95%):")
    print(f"   • Media:              {np.mean(var_vals):.6f}")
    print(f"   • Mediana:            {np.median(var_vals):.6f}")
    print(f"   • Peor caso:          {np.min(var_vals):.6f}")
    
    # Estadísticas de CVaR
    cvar_vals = list(inputs_modelo['cvar'].values())
    print(f"\n📉 Conditional VaR (CVaR 95%):")
    print(f"   • Media:              {np.mean(cvar_vals):.6f}")
    print(f"   • Mediana:            {np.median(cvar_vals):.6f}")
    print(f"   • Peor caso:          {np.min(cvar_vals):.6f}")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    from utils.CargarDatos import descargar_sp500_mensual
    
    # Descargar datos
    print("Descargando datos...")
    df_datos = descargar_sp500_mensual(start_year=2000, end_year=2015, guardar_csv=False)
    
    # Generar inputs (limitando a 50 activos para ejemplo rápido)
    inputs = generar_inputs_modelo(
        df_datos, 
        alpha_ewma=0.94,
        max_activos=50,  # Cambiar a None para procesar todos
        guardar_pickle=True,
        nombre_archivo='inputs_modelo.pkl'
    )
    
    # Ver resumen
    resumen_inputs(inputs)
    
    # Ejemplo de cómo cargar después
    # inputs_cargados = cargar_inputs_modelo('inputs_modelo.pkl')
