"""
BACKTESTING.PY - Comparación de rendimiento de portafolios vs S&P 500

Compara el desempeño de:
- Portafolio 1: Modelo Estocástico
- Portafolio 2: Modelo Robust Optimization
- Benchmark: S&P 500 (Buy and Hold)

Uso:
    python Backtesting.py
    
O personalizado:
    from Backtesting import ejecutar_backtesting
    ejecutar_backtesting(
        corrida_1='Corridas/20251025_150353_analisis_sp500',
        corrida_2='Corridas/20251025_160000_robust',
        año_inicio=2024,
        guardar_en='Corridas/Backtesting'
    )
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import pyomo.environ as pyo


def cargar_portafolio(ruta_corrida, nombre_modelo='ModeloEstocastico', nombre_archivo='resultados.xlsx'):
    """
    Carga los pesos del portafolio óptimo desde una corrida.
    
    Parámetros:
        ruta_corrida : str, ruta a la carpeta de la corrida
        nombre_modelo : str, nombre descriptivo del modelo
        nombre_archivo : str, nombre del archivo Excel con resultados
    
    Retorna:
        dict con:
            - pesos: Series con ticker y peso
            - nombre: nombre del modelo
            - ruta: ruta de la corrida
    """
    # Cargar inputs_modelo para obtener acciones
    with open(os.path.join(ruta_corrida, 'inputs_modelo.pkl'), 'rb') as f:
        inputs_modelo = pickle.load(f)
    
    # Cargar Excel de resultados para obtener pesos
    excel_path = os.path.join(ruta_corrida, nombre_archivo)
    df_acciones = pd.read_excel(excel_path, sheet_name='Acciones_Seleccionadas')
    
    # Crear Series de pesos
    pesos = pd.Series(
        df_acciones['Peso_W'].values,
        index=df_acciones['Ticker'].values
    )
    
    return {
        'pesos': pesos,
        'nombre': nombre_modelo,
        'ruta': ruta_corrida,
        'n_acciones': len(pesos),
        'tickers': pesos.index.tolist()
    }


def descargar_precios_backtesting(tickers, fecha_inicio, fecha_fin=None):
    """
    Descarga precios históricos para backtesting.
    
    Parámetros:
        tickers : list de tickers a descargar
        fecha_inicio : str o datetime, fecha de inicio (formato 'YYYY-MM-DD')
        fecha_fin : str o datetime, fecha final (default: hoy)
    
    Retorna:
        DataFrame con precios de cierre ajustados (index=Date, columns=Ticker)
    """
    if fecha_fin is None:
        fecha_fin = datetime.now()
    
    # Asegurar que tickers es una lista
    if isinstance(tickers, pd.Series):
        tickers = tickers.tolist()
    elif not isinstance(tickers, list):
        tickers = list(tickers)
    
    print(f"\n📥 Descargando precios históricos...")
    print(f"   Período: {fecha_inicio} a {fecha_fin}")
    print(f"   Tickers: {len(tickers)}")
    
    # Descargar precios diarios
    data = yf.download(
        tickers=tickers,
        start=fecha_inicio,
        end=fecha_fin,
        progress=False,
        threads=True
    )
    
    # Extraer precios de cierre ajustados
    if len(tickers) == 1:
        precios = data['Close'].to_frame()
        precios.columns = tickers
    else:
        precios = data['Close']
    
    # Rellenar NaN
    precios = precios.ffill().bfill()
    
    print(f"   ✅ {len(precios)} días descargados")
    
    return precios


def descargar_sp500_index(fecha_inicio, fecha_fin=None):
    """
    Descarga el índice S&P 500 (^GSPC) para benchmark.
    
    Parámetros:
        fecha_inicio : str o datetime
        fecha_fin : str o datetime
    
    Retorna:
        Series con precios del S&P 500
    """
    if fecha_fin is None:
        fecha_fin = datetime.now()
    
    print(f"\n📊 Descargando S&P 500 (^GSPC)...")
    
    sp500 = yf.download(
        '^GSPC',
        start=fecha_inicio,
        end=fecha_fin,
        progress=False
    )['Close']
    
    print(f"   ✅ {len(sp500)} días descargados")
    
    return sp500


def calcular_rendimiento_portafolio(precios, pesos):
    """
    Calcula el rendimiento acumulado de un portafolio con pesos fijos (buy and hold).
    
    Parámetros:
        precios : DataFrame con precios históricos (index=Date, columns=Ticker)
        pesos : Series con pesos del portafolio (index=Ticker, values=peso)
    
    Retorna:
        Series con rendimiento acumulado del portafolio
    """
    # Filtrar solo los tickers del portafolio
    tickers_portafolio = pesos.index.tolist()
    precios_portafolio = precios[tickers_portafolio]
    
    # Calcular rendimientos diarios logarítmicos
    rendimientos = np.log(precios_portafolio / precios_portafolio.shift(1)).dropna()
    
    # Alinear pesos con el orden de las columnas de rendimientos
    pesos_alineados = pesos.reindex(rendimientos.columns)
    
    # Rendimiento diario del portafolio (ponderado)
    rendimiento_diario = rendimientos.dot(pesos_alineados.values)
    
    # Rendimiento acumulado
    rendimiento_acumulado = np.exp(rendimiento_diario.cumsum()) - 1
    
    return rendimiento_acumulado


def calcular_rendimiento_sp500(precios_sp500):
    """
    Calcula el rendimiento acumulado del S&P 500.
    
    Parámetros:
        precios_sp500 : Series con precios del S&P 500
    
    Retorna:
        Series con rendimiento acumulado
    """
    # Normalizar a inicio = 0
    rendimiento_acumulado = (precios_sp500 / precios_sp500.iloc[0]) - 1
    
    return rendimiento_acumulado


def calcular_metricas_desempeno(rendimiento_acumulado, nombre='Portafolio'):
    """
    Calcula métricas de desempeño de un portafolio.
    
    Parámetros:
        rendimiento_acumulado : Series con rendimientos acumulados
        nombre : str, nombre del portafolio
    
    Retorna:
        dict con métricas
    """
    # Rendimiento total
    if isinstance(rendimiento_acumulado.iloc[-1], pd.Series):
        rendimiento_total = float(rendimiento_acumulado.iloc[-1].iloc[0])
    else:
        rendimiento_total = float(rendimiento_acumulado.iloc[-1])
    
    # Rendimientos diarios
    rendimientos_diarios = rendimiento_acumulado.pct_change().dropna()
    
    # Volatilidad anualizada (asumiendo 252 días de trading)
    vol_val = rendimientos_diarios.std() * np.sqrt(252)
    if isinstance(vol_val, pd.Series):
        volatilidad_anual = float(vol_val.iloc[0])
    else:
        volatilidad_anual = float(vol_val)
    
    # Sharpe Ratio (asumiendo tasa libre de riesgo = 0)
    if volatilidad_anual > 0:
        sharpe = rendimiento_total / volatilidad_anual
    else:
        sharpe = 0.0
    
    # Drawdown máximo
    valor_acumulado = 1 + rendimiento_acumulado
    max_anterior = valor_acumulado.cummax()
    drawdown = (valor_acumulado - max_anterior) / max_anterior
    dd_min = drawdown.min()
    if isinstance(dd_min, pd.Series):
        max_drawdown = float(dd_min.iloc[0])
    else:
        max_drawdown = float(dd_min)
    
    return {
        'nombre': nombre,
        'rendimiento_total': rendimiento_total,
        'volatilidad_anual': volatilidad_anual,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }


def graficar_comparacion(resultados, ruta_salida='backtesting_comparacion.png'):
    """
    Genera gráfico comparativo de los portafolios vs S&P 500.
    
    Parámetros:
        resultados : dict con resultados de backtesting
        ruta_salida : str, ruta donde guardar el gráfico
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Rendimiento acumulado
    ax1 = axes[0]
    
    for nombre, datos in resultados.items():
        rendimiento = datos['rendimiento_acumulado']
        
        # Asegurar que rendimiento es 1D
        if isinstance(rendimiento, pd.DataFrame):
            rendimiento = rendimiento.squeeze()
        
        ax1.plot(rendimiento.index, rendimiento * 100, label=nombre, linewidth=2)
    
    ax1.set_title('Comparación de Rendimientos Acumulados', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fecha', fontsize=12)
    ax1.set_ylabel('Rendimiento Acumulado (%)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    # Gráfico 2: Drawdown
    ax2 = axes[1]
    
    for nombre, datos in resultados.items():
        rendimiento = datos['rendimiento_acumulado']
        valor_acumulado = 1 + rendimiento
        max_anterior = valor_acumulado.cummax()
        drawdown = (valor_acumulado - max_anterior) / max_anterior
        
        # Asegurar que drawdown es 1D
        if isinstance(drawdown, pd.DataFrame):
            drawdown = drawdown.squeeze()
        
        ax2.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, label=nombre)
    
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    print(f"\n📊 Gráfico guardado: {ruta_salida}")
    
    return fig


def ejecutar_backtesting(corrida_1, corrida_2, 
                         nombre_archivo_1='resultados.xlsx',
                         nombre_archivo_2='resultados.xlsx',
                         año_inicio=2024, guardar_en=None):
    """
    Ejecuta backtesting completo comparando dos portafolios vs S&P 500.
    
    Parámetros:
        corrida_1 : str, ruta a la carpeta de la corrida del Modelo 1
        corrida_2 : str, ruta a la carpeta de la corrida del Modelo 2
        nombre_archivo_1 : str, nombre del archivo Excel del Modelo 1
        nombre_archivo_2 : str, nombre del archivo Excel del Modelo 2
        año_inicio : int, año de inicio del backtesting (default 2024)
        guardar_en : str, carpeta donde guardar resultados (default: Corridas/Backtesting)
    
    Retorna:
        dict con resultados completos
    """
    
    print("\n" + "="*80)
    print("🚀 INICIANDO BACKTESTING")
    print("="*80)
    
    # Configurar fechas
    fecha_inicio = f"{año_inicio}-01-01"
    fecha_fin = datetime.now()
    
    print(f"\nPeríodo: {fecha_inicio} a {fecha_fin.strftime('%Y-%m-%d')}")
    
    # 1. Cargar portafolios
    print("\n📁 Cargando portafolios...")
    portafolio_1 = cargar_portafolio(corrida_1, 'Portafolio 1 (Estocástico)', nombre_archivo_1)
    portafolio_2 = cargar_portafolio(corrida_2, 'Portafolio 2 (Robust)', nombre_archivo_2)
    
    print(f"   ✅ Portafolio 1: {portafolio_1['n_acciones']} acciones")
    print(f"   ✅ Portafolio 2: {portafolio_2['n_acciones']} acciones")
    
    # 2. Obtener todos los tickers únicos
    todos_tickers = list(set(portafolio_1['tickers'] + portafolio_2['tickers']))
    
    # 3. Descargar precios
    precios = descargar_precios_backtesting(todos_tickers, fecha_inicio, fecha_fin)
    sp500 = descargar_sp500_index(fecha_inicio, fecha_fin)
    
    # 4. Calcular rendimientos
    print("\n📈 Calculando rendimientos...")
    
    rend_portafolio_1 = calcular_rendimiento_portafolio(precios, portafolio_1['pesos'])
    rend_portafolio_2 = calcular_rendimiento_portafolio(precios, portafolio_2['pesos'])
    rend_sp500 = calcular_rendimiento_sp500(sp500)
    
    # Alinear fechas (usar intersección)
    fechas_comunes = rend_portafolio_1.index.intersection(rend_portafolio_2.index).intersection(rend_sp500.index)
    
    rend_portafolio_1 = rend_portafolio_1.loc[fechas_comunes]
    rend_portafolio_2 = rend_portafolio_2.loc[fechas_comunes]
    rend_sp500 = rend_sp500.loc[fechas_comunes]
    
    # 5. Calcular métricas
    print("\n📊 Calculando métricas de desempeño...")
    
    metricas_1 = calcular_metricas_desempeno(rend_portafolio_1, portafolio_1['nombre'])
    metricas_2 = calcular_metricas_desempeno(rend_portafolio_2, portafolio_2['nombre'])
    metricas_sp500 = calcular_metricas_desempeno(rend_sp500, 'S&P 500 (Buy & Hold)')
    
    # 6. Crear carpeta de salida
    if guardar_en is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        guardar_en = os.path.join('Corridas', f'Backtesting_{timestamp}')
    
    os.makedirs(guardar_en, exist_ok=True)
    
    # 7. Generar gráfico
    resultados_graficos = {
        portafolio_1['nombre']: {'rendimiento_acumulado': rend_portafolio_1},
        portafolio_2['nombre']: {'rendimiento_acumulado': rend_portafolio_2},
        'S&P 500 (Buy & Hold)': {'rendimiento_acumulado': rend_sp500}
    }
    
    ruta_grafico = os.path.join(guardar_en, 'comparacion_rendimientos.png')
    graficar_comparacion(resultados_graficos, ruta_grafico)
    
    # 8. Guardar métricas en Excel
    df_metricas = pd.DataFrame([metricas_1, metricas_2, metricas_sp500])
    
    ruta_excel = os.path.join(guardar_en, 'metricas_backtesting.xlsx')
    with pd.ExcelWriter(ruta_excel, engine='openpyxl') as writer:
        df_metricas.to_excel(writer, sheet_name='Metricas', index=False)
    
    print(f"\n💾 Métricas guardadas: {ruta_excel}")
    
    # 9. Imprimir resumen
    print("\n" + "="*80)
    print("📊 RESUMEN DE BACKTESTING")
    print("="*80)
    
    for metricas in [metricas_1, metricas_2, metricas_sp500]:
        print(f"\n{metricas['nombre']}:")
        print(f"   • Rendimiento Total:   {metricas['rendimiento_total']:>8.2%}")
        print(f"   • Volatilidad Anual:   {metricas['volatilidad_anual']:>8.2%}")
        print(f"   • Sharpe Ratio:        {metricas['sharpe_ratio']:>8.4f}")
        print(f"   • Max Drawdown:        {metricas['max_drawdown']:>8.2%}")
    
    print("\n" + "="*80)
    print(f"✅ Backtesting completado. Resultados en: {guardar_en}")
    print("="*80 + "\n")
    
    return {
        'portafolio_1': {'metricas': metricas_1, 'rendimientos': rend_portafolio_1},
        'portafolio_2': {'metricas': metricas_2, 'rendimientos': rend_portafolio_2},
        'sp500': {'metricas': metricas_sp500, 'rendimientos': rend_sp500},
        'ruta_salida': guardar_en
    }


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Configuración por defecto
    CORRIDA_1 = "Corridas/Corrida"  # Modelo Estocástico
    CORRIDA_2 = "Corridas/Corrida"  # Modelo Robust (ajustar nombre)
    AÑO_INICIO = 2024
    
    # Ejecutar backtesting
    resultados = ejecutar_backtesting(
        corrida_1=CORRIDA_1,
        corrida_2=CORRIDA_2,
        año_inicio=AÑO_INICIO
    )
