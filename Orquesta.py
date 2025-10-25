"""
ORQUESTA.PY - Pipeline de análisis cuantitativo para activos del S&P 500

Este script orquesta el pipeline completo:
1. Descarga datos históricos del S&P 500
2. Calcula rendimientos logarítmicos
3. Calcula matriz de covarianzas y desviaciones estándar
4. Ajusta distribuciones de probabilidad a cada activo
5. Calcula métricas de riesgo: P(R<0), VaR, CVaR
6. Genera reportes y datasets listos para modelado matemático

Uso:
    python Orquesta.py

Outputs:
    - resultados_activos.csv: métricas por activo
    - matriz_covarianzas.csv: matriz de covarianzas
    - rendimientos_historicos.csv: serie temporal de rendimientos
    - reporte_resumen.txt: resumen estadístico
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os

# Importar funciones desde CargarDatos.py
from CargarDatos import (
    descargar_sp500_mensual,
    proyectar_rendimientos,
    ajustar_distribucion_accion
)

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURACIÓN DEL PIPELINE
# ============================================================================

CONFIG = {
    'start_year': 2000,
    'end_year': 2015,
    'alpha_ewma': 0.94,
    'umbral_ajuste': 0.05,
    'nivel_confianza_var': 0.95,
    'max_activos_analizar': None,  # None = todos, o número específico
    'guardar_resultados': True,
    'dir_output': 'resultados_orquesta',
    'verbose_individual': False,  # True para ver análisis detallado de cada activo
}


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def crear_directorio_output(dir_path):
    """Crea el directorio de salida si no existe."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"✅ Directorio creado: {dir_path}")
    return dir_path


def calcular_matriz_correlacion(cov_matrix):
    """Calcula matriz de correlación a partir de covarianzas."""
    # Extraer desviaciones estándar de la diagonal
    std_devs = np.sqrt(np.diag(cov_matrix))
    # Crear matriz de correlación
    outer_product = np.outer(std_devs, std_devs)
    corr_matrix = cov_matrix / outer_product
    # Asegurar diagonal = 1 (por errores numéricos)
    np.fill_diagonal(corr_matrix, 1.0)
    return pd.DataFrame(corr_matrix, index=cov_matrix.index, columns=cov_matrix.columns)


def generar_reporte_resumen(resultados_df, config):
    """Genera reporte de texto con resumen estadístico."""
    reporte = []
    reporte.append("="*80)
    reporte.append("REPORTE RESUMEN - ANÁLISIS CUANTITATIVO S&P 500")
    reporte.append("="*80)
    reporte.append(f"\nFecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append(f"Período analizado: {config['start_year']} - {config['end_year']}")
    reporte.append(f"Alpha EWMA: {config['alpha_ewma']}")
    reporte.append(f"Nivel confianza VaR: {config['nivel_confianza_var']*100:.0f}%")
    
    reporte.append("\n" + "="*80)
    reporte.append("ESTADÍSTICAS GENERALES")
    reporte.append("="*80)
    reporte.append(f"Total de activos analizados: {len(resultados_df)}")
    reporte.append(f"\nDistribuciones ajustadas:")
    dist_counts = resultados_df['Mejor_Distribucion'].value_counts()
    for dist, count in dist_counts.items():
        reporte.append(f"  • {dist}: {count} ({count/len(resultados_df)*100:.1f}%)")
    
    reporte.append(f"\nMétodos utilizados:")
    metodo_counts = resultados_df['Metodo'].value_counts()
    for metodo, count in metodo_counts.items():
        reporte.append(f"  • {metodo}: {count} ({count/len(resultados_df)*100:.1f}%)")
    
    reporte.append("\n" + "="*80)
    reporte.append("RENDIMIENTOS")
    reporte.append("="*80)
    reporte.append(f"Rendimiento promedio:     {resultados_df['Rendimiento_Medio'].mean():.6f}")
    reporte.append(f"Rendimiento mediano:      {resultados_df['Rendimiento_Medio'].median():.6f}")
    reporte.append(f"Mejor rendimiento:        {resultados_df['Rendimiento_Medio'].max():.6f} ({resultados_df.loc[resultados_df['Rendimiento_Medio'].idxmax(), 'Ticker']})")
    reporte.append(f"Peor rendimiento:         {resultados_df['Rendimiento_Medio'].min():.6f} ({resultados_df.loc[resultados_df['Rendimiento_Medio'].idxmin(), 'Ticker']})")
    
    reporte.append("\n" + "="*80)
    reporte.append("VOLATILIDAD")
    reporte.append("="*80)
    reporte.append(f"Volatilidad promedio:     {resultados_df['Volatilidad'].mean():.6f}")
    reporte.append(f"Volatilidad mediana:      {resultados_df['Volatilidad'].median():.6f}")
    reporte.append(f"Mayor volatilidad:        {resultados_df['Volatilidad'].max():.6f} ({resultados_df.loc[resultados_df['Volatilidad'].idxmax(), 'Ticker']})")
    reporte.append(f"Menor volatilidad:        {resultados_df['Volatilidad'].min():.6f} ({resultados_df.loc[resultados_df['Volatilidad'].idxmin(), 'Ticker']})")
    
    reporte.append("\n" + "="*80)
    reporte.append("PROBABILIDAD DE RENDIMIENTO NEGATIVO")
    reporte.append("="*80)
    reporte.append(f"P(R<0) promedio:          {resultados_df['Prob_Rend_Negativo'].mean():.4f} ({resultados_df['Prob_Rend_Negativo'].mean()*100:.2f}%)")
    reporte.append(f"P(R<0) mediana:           {resultados_df['Prob_Rend_Negativo'].median():.4f} ({resultados_df['Prob_Rend_Negativo'].median()*100:.2f}%)")
    reporte.append(f"Mayor P(R<0):             {resultados_df['Prob_Rend_Negativo'].max():.4f} ({resultados_df.loc[resultados_df['Prob_Rend_Negativo'].idxmax(), 'Ticker']})")
    reporte.append(f"Menor P(R<0):             {resultados_df['Prob_Rend_Negativo'].min():.4f} ({resultados_df.loc[resultados_df['Prob_Rend_Negativo'].idxmin(), 'Ticker']})")
    
    reporte.append("\n" + "="*80)
    reporte.append(f"VALUE AT RISK (VaR) - {config['nivel_confianza_var']*100:.0f}% CONFIANZA")
    reporte.append("="*80)
    reporte.append(f"VaR promedio:             {resultados_df['VaR_95'].mean():.6f}")
    reporte.append(f"VaR mediano:              {resultados_df['VaR_95'].median():.6f}")
    reporte.append(f"Peor VaR (mayor pérdida): {resultados_df['VaR_95'].min():.6f} ({resultados_df.loc[resultados_df['VaR_95'].idxmin(), 'Ticker']})")
    reporte.append(f"Mejor VaR:                {resultados_df['VaR_95'].max():.6f} ({resultados_df.loc[resultados_df['VaR_95'].idxmax(), 'Ticker']})")
    
    reporte.append("\n" + "="*80)
    reporte.append(f"CONDITIONAL VAR (CVaR/ES) - {config['nivel_confianza_var']*100:.0f}% CONFIANZA")
    reporte.append("="*80)
    reporte.append(f"CVaR promedio:            {resultados_df['CVaR_95'].mean():.6f}")
    reporte.append(f"CVaR mediano:             {resultados_df['CVaR_95'].median():.6f}")
    reporte.append(f"Peor CVaR:                {resultados_df['CVaR_95'].min():.6f} ({resultados_df.loc[resultados_df['CVaR_95'].idxmin(), 'Ticker']})")
    reporte.append(f"Mejor CVaR:               {resultados_df['CVaR_95'].max():.6f} ({resultados_df.loc[resultados_df['CVaR_95'].idxmax(), 'Ticker']})")
    
    reporte.append("\n" + "="*80)
    reporte.append("TOP 10 ACTIVOS POR SHARPE RATIO (aprox)")
    reporte.append("="*80)
    # Sharpe aproximado = Rendimiento / Volatilidad (sin tasa libre de riesgo)
    resultados_df['Sharpe_Aprox'] = resultados_df['Rendimiento_Medio'] / resultados_df['Volatilidad']
    top_sharpe = resultados_df.nlargest(10, 'Sharpe_Aprox')[['Ticker', 'Rendimiento_Medio', 'Volatilidad', 'Sharpe_Aprox']]
    for idx, row in top_sharpe.iterrows():
        reporte.append(f"{row['Ticker']:6s} | Rend: {row['Rendimiento_Medio']:8.4f} | Vol: {row['Volatilidad']:8.4f} | Sharpe: {row['Sharpe_Aprox']:8.2f}")
    
    reporte.append("\n" + "="*80)
    reporte.append("FIN DEL REPORTE")
    reporte.append("="*80)
    
    return "\n".join(reporte)


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def ejecutar_pipeline(config=CONFIG):
    """
    Ejecuta el pipeline completo de análisis cuantitativo.
    
    Pasos:
        1. Descarga datos S&P 500
        2. Calcula rendimientos y matriz de covarianzas (EWMA)
        3. Analiza distribución de cada activo
        4. Calcula métricas de riesgo (VaR, CVaR, P(R<0))
        5. Guarda resultados
    
    Returns:
        dict con todos los resultados generados
    """
    
    print("\n" + "="*80)
    print("🚀 INICIANDO PIPELINE DE ANÁLISIS CUANTITATIVO")
    print("="*80 + "\n")
    
    # --- PASO 1: Descargar datos ---
    print("📥 PASO 1: Descargando datos del S&P 500...")
    print(f"   Período: {config['start_year']} - {config['end_year']}")
    
    df_datos = descargar_sp500_mensual(
        start_year=config['start_year'],
        end_year=config['end_year'],
        guardar_csv=False
    )
    
    tickers_disponibles = df_datos['Ticker'].unique()
    print(f"   ✅ Descargados {len(df_datos)} registros de {len(tickers_disponibles)} tickers")
    
    # --- PASO 2: Calcular rendimientos y covarianzas (EWMA) ---
    print("\n📊 PASO 2: Calculando rendimientos y matriz de covarianzas (EWMA)...")
    print(f"   Alpha EWMA: {config['alpha_ewma']}")
    
    mu_pred, cov_ewma = proyectar_rendimientos(
        df_datos,
        alpha=config['alpha_ewma'],
        nan_threshold=0.1
    )
    
    # Calcular matriz de correlación
    corr_matrix = calcular_matriz_correlacion(cov_ewma)
    
    # Calcular desviaciones estándar
    desv_std = pd.Series(np.sqrt(np.diag(cov_ewma)), index=cov_ewma.index)
    
    print(f"   ✅ Matriz de covarianzas: {cov_ewma.shape}")
    print(f"   ✅ Rendimientos esperados calculados para {len(mu_pred)} activos")
    
    # --- PASO 3: Analizar distribución de cada activo ---
    print("\n🔬 PASO 3: Analizando distribución de probabilidad de cada activo...")
    
    # Limitar cantidad de activos si se especificó
    tickers_analizar = mu_pred.index.tolist()
    if config['max_activos_analizar'] is not None:
        tickers_analizar = tickers_analizar[:config['max_activos_analizar']]
        print(f"   ℹ️ Limitando análisis a {config['max_activos_analizar']} activos")
    
    resultados_activos = []
    errores = []
    
    total = len(tickers_analizar)
    for i, ticker in enumerate(tickers_analizar, 1):
        try:
            print(f"   [{i}/{total}] Analizando {ticker}...", end='\r')
            
            resultado = ajustar_distribucion_accion(
                df_datos,
                ticker,
                verbose=config['verbose_individual'],
                umbral_ajuste=config['umbral_ajuste'],
                nivel_confianza=config['nivel_confianza_var']
            )
            
            # Extraer métricas clave
            resultados_activos.append({
                'Ticker': ticker,
                'Rendimiento_Medio': resultado['estadisticas']['media'],
                'Volatilidad': resultado['estadisticas']['desviacion'],
                'Skewness': resultado['estadisticas']['skewness'],
                'Kurtosis': resultado['estadisticas']['kurtosis'],
                'N_Observaciones': resultado['estadisticas']['n_observaciones'],
                'Mejor_Distribucion': resultado['mejor_distribucion'],
                'Metodo': resultado['metodo_usado'],
                'Ajuste_Aceptable': resultado['ajuste_aceptable'],
                'KS_pvalue': resultado['ks_pvalue'],
                'Prob_Rend_Negativo': resultado['prob_rendimiento_negativo'],
                'Prob_Empirica': resultado['prob_empirica'],
                'VaR_95': resultado['var'],
                'CVaR_95': resultado['cvar'],
                'IC_Empirico_Lower': resultado['ic_empirico'][0],
                'IC_Empirico_Upper': resultado['ic_empirico'][1],
            })
            
        except Exception as e:
            errores.append({'Ticker': ticker, 'Error': str(e)})
            print(f"   ⚠️ Error en {ticker}: {str(e)[:50]}")
    
    print(f"\n   ✅ Análisis completado: {len(resultados_activos)} exitosos, {len(errores)} errores")
    
    # Convertir a DataFrame
    df_resultados = pd.DataFrame(resultados_activos)
    df_errores = pd.DataFrame(errores) if errores else None
    
    # --- PASO 4: Generar reporte resumen ---
    print("\n📝 PASO 4: Generando reporte resumen...")
    reporte_texto = generar_reporte_resumen(df_resultados, config)
    print(reporte_texto)
    
    # --- PASO 5: Guardar resultados ---
    if config['guardar_resultados']:
        print("\n💾 PASO 5: Guardando resultados...")
        dir_output = crear_directorio_output(config['dir_output'])
        
        # Guardar CSVs
        path_resultados = os.path.join(dir_output, 'resultados_activos.csv')
        df_resultados.to_csv(path_resultados, index=False)
        print(f"   ✅ {path_resultados}")
        
        path_covarianzas = os.path.join(dir_output, 'matriz_covarianzas.csv')
        cov_ewma.to_csv(path_covarianzas)
        print(f"   ✅ {path_covarianzas}")
        
        path_correlacion = os.path.join(dir_output, 'matriz_correlacion.csv')
        corr_matrix.to_csv(path_correlacion)
        print(f"   ✅ {path_correlacion}")
        
        path_rendimientos = os.path.join(dir_output, 'rendimientos_esperados.csv')
        mu_pred.to_csv(path_rendimientos, header=['Rendimiento_Esperado'])
        print(f"   ✅ {path_rendimientos}")
        
        path_desvios = os.path.join(dir_output, 'desviaciones_estandar.csv')
        desv_std.to_csv(path_desvios, header=['Desviacion_Estandar'])
        print(f"   ✅ {path_desvios}")
        
        # Guardar reporte de texto
        path_reporte = os.path.join(dir_output, 'reporte_resumen.txt')
        with open(path_reporte, 'w', encoding='utf-8') as f:
            f.write(reporte_texto)
        print(f"   ✅ {path_reporte}")
        
        # Guardar errores si hubo
        if df_errores is not None:
            path_errores = os.path.join(dir_output, 'errores.csv')
            df_errores.to_csv(path_errores, index=False)
            print(f"   ⚠️ {path_errores}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*80 + "\n")
    
    # Retornar todos los resultados
    return {
        'df_datos': df_datos,
        'df_resultados': df_resultados,
        'df_errores': df_errores,
        'mu_pred': mu_pred,
        'cov_ewma': cov_ewma,
        'corr_matrix': corr_matrix,
        'desv_std': desv_std,
        'reporte_texto': reporte_texto,
        'config': config
    }


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Ejecutar pipeline con configuración por defecto
    resultados = ejecutar_pipeline(CONFIG)
    
    # Opcional: Análisis adicionales
    print("\n💡 Sugerencias de uso:")
    print("   - Los archivos CSV están listos para modelado matemático")
    print("   - Use matriz_covarianzas.csv para optimización de portafolio")
    print("   - Use resultados_activos.csv para selección de activos")
    print("   - VaR y CVaR pueden usarse para restricciones de riesgo")
    print("\n   Ejemplo de acceso a resultados:")
    print("   >>> resultados['df_resultados'].head()")
    print("   >>> resultados['cov_ewma']")
    print("   >>> resultados['mu_pred']")
