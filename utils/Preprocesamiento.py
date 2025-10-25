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
        df_datos : DataFrame con columnas ['Date', 'Ticker', 'Close']
        alpha_ewma : float, parámetro EWMA (default 0.94)
        max_activos : int, límite de activos a procesar (None = todos)
        guardar_pickle : bool, si True guarda el diccionario en pickle
        nombre_archivo : str, nombre del archivo pickle
    
    Retorna:
        dict con estructura compatible con ModeloEstocastico.py:
            - set_acciones: list de tickers
            - rendimiento_esperado: dict {ticker: mu}
            - desvio_estandar: dict {ticker: sigma}
            - covarianzas: dict {(ticker_i, ticker_j): cov}
            - probabilidad_perdida: dict {ticker: P(R<0)}
            - var: dict {ticker: VaR al 95%}
            - cvar: dict {ticker: CVaR al 95%}
            - metadata: dict con información de la corrida
    """
    
    print("\n📊 Generando inputs para modelo estocástico...")
    
    # 1. Calcular rendimientos y covarianzas EWMA
    print("   Calculando EWMA...")
    mu_pred, cov_ewma = proyectar_rendimientos(df_datos, alpha=alpha_ewma)
    
    # 2. Seleccionar activos
    tickers = mu_pred.index.tolist()
    if max_activos:
        tickers = tickers[:max_activos]
        mu_pred = mu_pred.loc[tickers]
        cov_ewma = cov_ewma.loc[tickers, tickers]
    
    print(f"   Procesando {len(tickers)} activos...")
    
    # 3. Calcular volatilidades (desviaciones estándar)
    desvio_estandar = {ticker: np.sqrt(cov_ewma.loc[ticker, ticker]) for ticker in tickers}
    
    # 4. Analizar distribuciones y calcular métricas de riesgo
    probabilidad_perdida = {}
    var_dict = {}
    cvar_dict = {}
    
    errores = []
    for i, ticker in enumerate(tickers, 1):
        print(f"   [{i}/{len(tickers)}] {ticker}...", end='\r')
        try:
            resultado = ajustar_distribucion_accion(
                df_datos, 
                ticker, 
                verbose=False,
                umbral_ajuste=0.05,
                nivel_confianza=0.95
            )
            probabilidad_perdida[ticker] = resultado['prob_rendimiento_negativo']
            var_dict[ticker] = resultado['var']
            cvar_dict[ticker] = resultado['cvar']
        except Exception as e:
            errores.append(ticker)
            # Valores por defecto si falla el análisis
            probabilidad_perdida[ticker] = 0.5
            var_dict[ticker] = -2 * desvio_estandar[ticker]
            cvar_dict[ticker] = -3 * desvio_estandar[ticker]
    
    print(f"\n   ✅ Análisis completado ({len(errores)} errores)")
    
    # 5. Convertir matriz de covarianzas a diccionario
    covarianzas = {(i, j): cov_ewma.loc[i, j] 
                   for i in tickers for j in tickers}
    
    # 6. Construir diccionario de inputs (nombres compatibles con ModeloEstocastico.py)
    inputs_modelo = {
        'set_acciones': tickers,
        'rendimiento_esperado': mu_pred.to_dict(),
        'desvio_estandar': desvio_estandar,
        'covarianzas': covarianzas,
        'probabilidad_perdida': probabilidad_perdida,
        'var': var_dict,
        'cvar': cvar_dict,
        'metadata': {
            'n_activos': len(tickers),
            'alpha_ewma': alpha_ewma,
            'activos_con_error': errores,
            'fecha_generacion': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    # 7. Guardar pickle si se solicita
    if guardar_pickle:
        with open(nombre_archivo, 'wb') as f:
            pickle.dump(inputs_modelo, f)
        print(f"   💾 Archivo guardado: {nombre_archivo}")
    
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
    print(f"   • Alpha EWMA:         {meta['alpha_ewma']}")
    print(f"   • Fecha generación:   {meta['fecha_generacion']}")
    print(f"   • Errores:            {len(meta['activos_con_error'])}")
    
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
