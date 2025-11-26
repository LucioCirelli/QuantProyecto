"""
ORQUESTA.PY - Pipeline simplificado de análisis cuantitativo S&P 500

Uso:
    python Orquesta.py
    
Genera carpeta: Corridas/[timestamp]_[nombre]/
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from utils.CargarDatos import descargar_sp500_mensual, proyectar_rendimientos
from utils.Preprocesamiento import generar_inputs_modelo, resumen_inputs


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# ============================================================================
# PIPELINE
# ============================================================================

def ejecutar_preprocesamiento(nombre_corrida, start_year, end_year, alpha_ewma, max_activos = None):
    """Pipeline simplificado de análisis."""
    
    # Crear carpeta de corrida
    dir_corrida = os.path.join('Corridas', f'{nombre_corrida}')
    os.makedirs(dir_corrida, exist_ok=True)
    
    print(f"\n🚀 Corrida: {dir_corrida}\n")
    
    # 1. Descargar datos
    print("📥 Descargando datos S&P 500...")
    df_datos = descargar_sp500_mensual(start_year, end_year, guardar_csv=False)
    print(f"✅ {len(df_datos['Ticker'].unique())} tickers descargados")
    
    # 2. Calcular rendimientos y covarianzas
    print("\n📊 Calculando rendimientos y covarianzas...")
    mu_pred, cov_ewma = proyectar_rendimientos(df_datos, alpha=alpha_ewma)
    desv_std = pd.Series(np.sqrt(np.diag(cov_ewma)), index=cov_ewma.index)
    print(f"✅ Matriz de covarianzas: {cov_ewma.shape}")
    
    # 3. Generar inputs para modelo estocástico
    print("\n🔬 Generando inputs para modelo estocástico...")
    inputs_modelo = generar_inputs_modelo(
        df_datos,
        alpha_ewma=alpha_ewma,
        max_activos=max_activos,
        guardar_pickle=False  # Se guardará en la carpeta de corrida
    )
    print(f"✅ Inputs generados para {len(inputs_modelo['set_acciones'])} activos")
    
    # 4. Guardar archivos
    print("\n💾 Guardando archivos...")
    
    # Matrices y rendimientos
    cov_ewma.to_csv(os.path.join(dir_corrida, 'matriz_covarianzas.csv'))
    mu_pred.to_csv(os.path.join(dir_corrida, 'rendimientos_esperados.csv'))
    desv_std.to_csv(os.path.join(dir_corrida, 'desviaciones_estandar.csv'))
    
    # Inputs del modelo (pickle)
    import pickle
    with open(os.path.join(dir_corrida, 'inputs_modelo.pkl'), 'wb') as f:
        pickle.dump(inputs_modelo, f)
    
    # Resumen de inputs
    print("\n📋 Resumen de inputs generados:")
    resumen_inputs(inputs_modelo)
    
    print(f"✅ Archivos guardados en: {dir_corrida}\n")
    
    return inputs_modelo, mu_pred, cov_ewma, desv_std


if __name__ == "__main__":
    nombre_corrida = "Corrida"
    start_year = 2010
    end_year = 2023
    alpha_ewma = 0.94
    max_activos = None  # None = todos
    ejecutar_preprocesamiento(nombre_corrida, start_year, end_year, alpha_ewma, max_activos)
