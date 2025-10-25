"""
POSTPROCESAMIENTO.PY - Generación de reportes de resultados del modelo

Genera archivo Excel simple con resultados del modelo.

Uso:
    from Postprocesamiento import generar_reporte_excel
    generar_reporte_excel(model, inputs_modelo, 'resultados.xlsx')
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo


def generar_reporte_excel(model, inputs_modelo, nombre_archivo='resultados.xlsx'):
    """
    Genera reporte Excel simple con resultados del modelo.
    
    Parámetros:
        model : modelo Pyomo resuelto
        inputs_modelo : dict con inputs originales
        nombre_archivo : str, nombre del archivo Excel de salida
    
    Retorna:
        str : ruta del archivo generado
    """
    
    print(f"\n📊 Generando reporte Excel: {nombre_archivo}")
    
    # 1. Extraer acciones seleccionadas (con peso > 0)
    acciones_seleccionadas = []
    
    for accion in model.ACCION:
        w = pyo.value(model.W[accion])
        if w > 1e-6:  # Solo pesos significativos
            acciones_seleccionadas.append({
                'Ticker': accion,
                'Peso_W': w,
                'Rendimiento_Esperado': pyo.value(model.mu[accion]),
                'Desvio_Estandar': pyo.value(model.desvio[accion]),
                'VaR_95': pyo.value(model.var[accion]),
                'CVaR_95': pyo.value(model.cvar[accion]),
                'Prob_Perdida': pyo.value(model.probabilidad_perdida[accion])
            })
    
    df_seleccionadas = pd.DataFrame(acciones_seleccionadas)
    df_seleccionadas = df_seleccionadas.sort_values('Peso_W', ascending=False)
    
    # 2. Métricas del portafolio
    rendimiento_total = pyo.value(model.RENDIMIENTO_PORTAFOLIO)
    riesgo_total = pyo.value(model.RIESGO_PORTAFOLIO)
    volatilidad = np.sqrt(riesgo_total)
    costo_perdida = pyo.value(model.COSTO_PERDIDA)
    valor_objetivo = pyo.value(list(model.component_objects(pyo.Objective))[0])
    
    # Sharpe ratio
    exceso_rendimiento = rendimiento_total - model.tasa_libre_riesgo
    sharpe_ratio = exceso_rendimiento / volatilidad if volatilidad > 0 else 0
    
    df_metricas = pd.DataFrame([
        {'Metrica': 'Rendimiento_Esperado', 'Valor': rendimiento_total},
        {'Metrica': 'Volatilidad', 'Valor': volatilidad},
        {'Metrica': 'Sharpe_Ratio', 'Valor': sharpe_ratio},
        {'Metrica': 'Riesgo_Varianza', 'Valor': riesgo_total},
        {'Metrica': 'CVaR_Portafolio', 'Valor': costo_perdida},
        {'Metrica': 'Funcion_Objetivo', 'Valor': valor_objetivo},
        {'Metrica': 'N_Acciones', 'Valor': len(df_seleccionadas)},
        {'Metrica': 'Peso_Total_Acciones', 'Valor': df_seleccionadas['Peso_W'].sum()}
    ])
    
    # 3. Parámetros del modelo
    df_parametros = pd.DataFrame([
        {'Parametro': 'max_acciones', 'Valor': model.max_acciones},
        {'Parametro': 'w_minimo', 'Valor': model.w_minimo},
        {'Parametro': 'w_maximo', 'Valor': model.w_maximo},
        {'Parametro': 'rendimiento_minimo', 'Valor': model.rendimiento_minimo_portafolio},
        {'Parametro': 'tasa_libre_riesgo', 'Valor': model.tasa_libre_riesgo}
    ])
    
    # 4. Guardar Excel
    with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
        df_metricas.to_excel(writer, sheet_name='Metricas', index=False)
        df_seleccionadas.to_excel(writer, sheet_name='Acciones_Seleccionadas', index=False)
        df_parametros.to_excel(writer, sheet_name='Parametros', index=False)
    
    print(f"✅ Reporte generado: {nombre_archivo}")
    print(f"\n📋 Resumen:")
    print(f"   • Acciones seleccionadas: {len(df_seleccionadas)}")
    print(f"   • Rendimiento esperado:   {rendimiento_total:.4%}")
    print(f"   • Volatilidad:            {volatilidad:.4%}")
    print(f"   • Sharpe Ratio:           {sharpe_ratio:.4f}")
    
    return nombre_archivo


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("Para usar este script:")
    print("1. Ejecuta ModeloEstocastico.py")
    print("2. El reporte se genera automáticamente en la carpeta de la corrida")

