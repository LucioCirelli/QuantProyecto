"""
FUNCIONES_BACKTESTING.PY - Lógica completa de backtesting

- Rebalanceo mensual con ventana móvil
- Cálculo de inputs (mu, Sigma, delta)
- Ejecución de modelos
- Cálculo de métricas
- Generación de gráficos

"""

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# WRAPPERS DE MODELOS
# ============================================================================

def get_modelo_wrapper(nombre_modelo, parametros):
    """
    Retorna una función wrapper para el modelo especificado.
    
    La función retornada tiene signature:
        wrapper(mu_dict, Sigma_dict, delta_dict, cvar_dict, var_dict, prob_perdida_dict, pesos_anteriores) -> dict de pesos
    """
    
    if nombre_modelo == 'minimizador_riesgo':
        def wrapper(mu_dict, Sigma_dict, delta_dict, cvar_dict=None, var_dict=None, prob_perdida_dict=None, pesos_anteriores=None):
            from utils_backend.ModeloMinimizadorRiesgo import modelo_minimizador_riesgo
            
            try:
                return modelo_minimizador_riesgo(
                    mu_dict=mu_dict,
                    Sigma_dict=Sigma_dict,
                    delta_dict=delta_dict,
                    cvar_dict=cvar_dict,
                    var_dict=var_dict,
                    prob_perdida_dict=prob_perdida_dict,
                    max_acciones=parametros.get('max_acciones', 10),
                    w_minimo=parametros.get('w_minimo', 0.05),
                    w_maximo=parametros.get('w_maximo', 0.3),
                    rendimiento_minimo=parametros.get('rendimiento_minimo', np.log(1.015))
                )
                
            except Exception as e:
                print(f"Error en minimizador_riesgo: {e}")
                return {}
        
        return wrapper
    
    elif nombre_modelo == 'maximizador_beneficio':
        def wrapper(mu_dict, Sigma_dict, delta_dict, cvar_dict=None, var_dict=None, prob_perdida_dict=None, pesos_anteriores=None):
            from utils_backend.ModeloMaximizadorBeneficio import modelo_maximizador_beneficio
            
            try:
                return modelo_maximizador_beneficio(
                    mu_dict=mu_dict,
                    Sigma_dict=Sigma_dict,
                    delta_dict=delta_dict,
                    cvar_dict=cvar_dict,
                    var_dict=var_dict,
                    prob_perdida_dict=prob_perdida_dict,
                    max_acciones=parametros.get('max_acciones', 10),
                    w_minimo=parametros.get('w_minimo', 0.05),
                    w_maximo=parametros.get('w_maximo', 0.3),
                    rendimiento_minimo=parametros.get('rendimiento_minimo', np.log(1.015))
                )
                
            except Exception as e:
                print(f"Error en maximizador_beneficio: {e}")
                return {}
        
        return wrapper
    
    elif nombre_modelo == 'mvo':
        def wrapper(mu_dict, Sigma_dict, delta_dict, cvar_dict=None, var_dict=None, prob_perdida_dict=None, pesos_anteriores=None):
            
            from utils_backend.OptimizarCartera import minimizar_riesgo_custom
            
            try:
                return minimizar_riesgo_custom(
                    mu_dict=mu_dict,
                    Sigma_dict=Sigma_dict,
                    delta_dict=delta_dict,
                    aversion=parametros.get('aversion_riesgo', 2),
                    size_portfolio_max=parametros.get('max_activos', 30),
                    size_portfolio_min=parametros.get('min_activos', 20),
                    peso_maximo=parametros.get('peso_maximo', 0.10),
                    peso_minimo=parametros.get('peso_minimo', 0.015)
                )
            except (ImportError, TypeError):
                # Fallback a versión original
                from utils_backend.OptimizarCartera import minimizar_riesgo
                return minimizar_riesgo(
                    mu_dict, Sigma_dict, delta_dict,
                    aversion=parametros.get('aversion_riesgo', 2)
                )
        
        return wrapper
    
    else:
        raise ValueError(f"Modelo desconocido: {nombre_modelo}")


# ============================================================================
# BACKTESTING COMPLETO
# ============================================================================

def ejecutar_backtesting_completo(df_tickers, df_spy, nombre_modelo, parametros, 
                                  start_date, end_date=None, window_meses=100, rebalance_freq=1,
                                  parametros_preprocesamiento=None):
    """
    Ejecuta backtesting completo con rebalanceo mensual.
    
    Args:
        start_date: Fecha de inicio del backtesting
        end_date: Fecha de fin del backtesting (None = hasta el final de los datos)
        window_meses: Ventana histórica en meses para calcular inputs
        parametros_preprocesamiento: dict con peso_media, peso_momentum, meses_momentum, nivel_confianza
    """
    
    # Parámetros de preprocesamiento por defecto
    if parametros_preprocesamiento is None:
        parametros_preprocesamiento = {
            'peso_media': 0.5,
            'peso_momentum': 0.5,
            'meses_momentum': 3,
            'nivel_confianza': 1.96
        }
    
    # Obtener wrapper del modelo (retorna diccionario de pesos)
    wrapper_modelo = get_modelo_wrapper(nombre_modelo, parametros)
    
    # Preparar datos
    df_tickers = df_tickers.copy()
    df_spy = df_spy.copy()
    df_tickers['Date'] = pd.to_datetime(df_tickers['Date'])
    df_spy['Date'] = pd.to_datetime(df_spy['Date'])
    df_tickers = df_tickers.sort_values(['Ticker', 'Date'])
    df_spy = df_spy.sort_values(['Ticker', 'Date'])
    
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date) if end_date else None
    
    # Obtener fechas de rebalanceo
    all_dates = sorted(df_tickers['Date'].unique())
    dates = []
    current_date = None
    for d in all_dates:
        # Aplicar filtro de fecha de fin si existe
        if end_date and d > end_date:
            break
        if d >= start_date:
            if current_date is None:
                dates.append(d)
                current_date = d
            elif d >= current_date + relativedelta(months=rebalance_freq):
                dates.append(d)
                current_date = d
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING: {nombre_modelo.upper()}")
    print(f"{'='*80}")
    print(f"Período: {dates[0].date()} a {dates[-1].date()}")
    print(f"Rebalanceos: {len(dates)}")
    print(f"{'='*80}\n")
    
    results = []
    benchmark_results = []
    portfolio_weights = []
    pesos_anteriores = None
    
    # LOOP DE REBALANCEO
    for idx, d in enumerate(dates):
        print(f"[{idx+1}/{len(dates)}] {d.date()}", end=" ")
        
        # Ventana de entrenamiento
        start_window = d - relativedelta(months=window_meses)
        df_window = df_tickers[
            (df_tickers['Date'] >= start_window) & 
            (df_tickers['Date'] < d)
        ]
        
        # Validar observaciones
        ticker_counts = df_window.groupby('Ticker').size()
        valid_tickers = ticker_counts[ticker_counts >= 12].index
        df_window = df_window[df_window['Ticker'].isin(valid_tickers)]
        
        if df_window.empty:
            print("⚠️ Sin datos")
            continue
        
        # Calcular inputs (MÉTODO PARAMETRIZABLE)
        mean_return = df_window.groupby('Ticker')['Return'].mean()
        
        # Momentum con N meses configurables
        meses_mom = parametros_preprocesamiento['meses_momentum']
        momentum = df_window.groupby('Ticker').apply(
            lambda x: x.sort_values('Date').tail(meses_mom)['Return'].mean()
        )
        
        # Combinar con pesos configurables
        w_media = parametros_preprocesamiento['peso_media']
        w_momentum = parametros_preprocesamiento['peso_momentum']
        mu = w_media * mean_return + w_momentum * momentum
        mu = mu.fillna(mu.median())
        
        # Matriz de covarianzas
        Sigma = df_window.pivot(index='Date', columns='Ticker', values='Return').cov()
        sigma_i = np.sqrt(np.diag(Sigma))
        T = len(df_window['Date'].unique())
        
        # Delta con nivel de confianza configurable
        z_score = parametros_preprocesamiento['nivel_confianza']
        delta = z_score * sigma_i / np.sqrt(T)
        
        # Limpiar NaNs
        mu = mu.replace([np.inf, -np.inf], np.nan).fillna(mu.median())
        sigma_i = np.nan_to_num(sigma_i, nan=sigma_i[np.isfinite(sigma_i)].mean())
        delta = np.nan_to_num(delta, nan=delta[np.isfinite(delta)].mean())
        
        # Convertir a dicts
        mu_dict = mu.to_dict()
        Sigma_dict = Sigma.stack().to_dict()
        delta_dict = pd.Series(delta, index=Sigma.index).to_dict()
        sigma_dict = pd.Series(sigma_i, index=Sigma.index).to_dict()
        
        # ====================================================================
        # CALCULAR CVaR, VaR Y PROBABILIDAD DE PÉRDIDA (para modelos Pyomo)
        # ====================================================================
        # CVaR al 95% de confianza: |mu + 2.06 * sigma|
        # Fórmula: CVaR_0.95 ≈ |μ + 2.06σ| para distribución normal
        cvar_dict = {ticker: abs(mu_dict[ticker] + 2.06 * sigma_dict[ticker]) 
                     for ticker in sigma_dict.keys()}
        
        # VaR al 95% de confianza: |mu + 1.65 * sigma|
        # Fórmula: VaR_0.95 = |μ + 1.65σ| para distribución normal
        var_dict = {ticker: abs(mu_dict[ticker] + 1.65 * sigma_dict[ticker]) 
                    for ticker in sigma_dict.keys()}
        
        # Probabilidad de pérdida: % de retornos negativos históricos
        prob_perdida_dict = {}
        for ticker in mu_dict.keys():
            ticker_returns = df_window[df_window['Ticker'] == ticker]['Return']
            if len(ticker_returns) > 0:
                prob_perdida_dict[ticker] = (ticker_returns < 0).mean()
            else:
                prob_perdida_dict[ticker] = 0.05  # Default 5%
        
        # Ejecutar modelo (wrapper retorna dict de pesos)
        try:
            pesos = wrapper_modelo(mu_dict, Sigma_dict, delta_dict, 
                                  cvar_dict=cvar_dict, var_dict=var_dict, 
                                  prob_perdida_dict=prob_perdida_dict,
                                  pesos_anteriores=pesos_anteriores)
            
            # # Validar que se obtuvieron pesos
            # if not pesos or all(v == 0 for v in pesos.values()):
            #     raise ValueError("Modelo no retornó pesos válidos")
            print(pesos)
            
            # Convertir dict de pesos a Series para operar con pandas
            w_opt = pd.Series(pesos).reindex(Sigma.columns).fillna(0)
            pesos_anteriores = w_opt.copy()
            
            weights_dict = {'Date': d}
            weights_dict.update(w_opt.to_dict())
            portfolio_weights.append(weights_dict)
            
            print(f"✓ {(w_opt > 0).sum()} activos", end=" ")
        except Exception as e:
            print(f"✗ Error: {e}")
            if pesos_anteriores is None:
                print(" (saltando período)")
                continue
            print(" (usando pesos anteriores)")
            w_opt = pesos_anteriores.copy()
        
        # Calcular retornos
        next_period = df_tickers[
            (df_tickers['Date'] >= d) &
            (df_tickers['Date'] < d + relativedelta(months=rebalance_freq))
        ]
        
        if next_period.empty:
            print("⚠️ Sin próximo período")
            continue
        
        next_returns = next_period.pivot(index='Date', columns='Ticker', values='Return').fillna(0)
        common_tickers = next_returns.columns.intersection(w_opt.index)
        next_returns = next_returns[common_tickers]
        w_opt_aligned = w_opt[common_tickers]
        
        if w_opt_aligned.sum() == 0:
            print("⚠️ Pesos = 0")
            continue
        
        w_opt_aligned = w_opt_aligned / w_opt_aligned.sum()
        
        port_ret = (next_returns.values @ w_opt_aligned.values).sum()
        results.append({'Date': d, 'Portfolio_Return': port_ret})
        
        # Benchmark
        mask_spy = (df_spy['Date'] >= d) & (df_spy['Date'] < d + relativedelta(months=rebalance_freq))
        bench_ret = df_spy.loc[mask_spy, 'Return'].sum()
        benchmark_results.append({'Date': d, 'Benchmark_Return': bench_ret})
        
        print(f"→ Port: {port_ret:+.4f} | Bench: {bench_ret:+.4f}")
    
    # Consolidar resultados
    df_results = pd.DataFrame(results).sort_values('Date')
    df_benchmark = pd.DataFrame(benchmark_results).sort_values('Date')
    df_plot = pd.merge(df_results, df_benchmark, on='Date', how='inner')
    
    df_plot['Cum_Portfolio'] = df_plot['Portfolio_Return'].cumsum()
    df_plot['Cum_Benchmark'] = df_plot['Benchmark_Return'].cumsum()
    df_plot['Cum_Portfolio_Pct'] = np.exp(df_plot['Cum_Portfolio']) - 1
    df_plot['Cum_Benchmark_Pct'] = np.exp(df_plot['Cum_Benchmark']) - 1
    
    print(df_plot)
    df_plot.to_clipboard()
    
    # Calcular métricas
    metricas = calcular_metricas(df_plot, rebalance_freq)
    
    print(f"\n{'='*80}")
    print(f"COMPLETADO: Retorno total = {metricas['total_return_port']*100:.2f}%")
    print(f"{'='*80}\n")
    
    return {
        'df_plot': df_plot,
        'portfolio_weights': portfolio_weights,
        'metricas': metricas,
        'nombre_modelo': nombre_modelo
    }


# ============================================================================
# CÁLCULO DE MÉTRICAS
# ============================================================================

def calcular_metricas(df_plot, rebalance_freq=1):
    """Calcula todas las métricas"""
    
    start_date = df_plot['Date'].iloc[0]
    end_date = df_plot['Date'].iloc[-1]
    years = (end_date - start_date).days / 365.25
    periods_per_year = 12 / rebalance_freq
    
    total_return_port = df_plot['Cum_Portfolio_Pct'].iloc[-1]
    total_return_bench = df_plot['Cum_Benchmark_Pct'].iloc[-1]
    
    ann_return_port = (1 + total_return_port) ** (1/years) - 1
    ann_return_bench = (1 + total_return_bench) ** (1/years) - 1
    
    vol_port = df_plot['Portfolio_Return'].std()
    vol_bench = df_plot['Benchmark_Return'].std()
    ann_vol_port = vol_port * np.sqrt(periods_per_year)
    ann_vol_bench = vol_bench * np.sqrt(periods_per_year)
    
    # Max Drawdown
    cumulative_wealth_port = 1 + df_plot['Cum_Portfolio_Pct']
    running_max_port = cumulative_wealth_port.expanding().max()
    drawdown_port = (cumulative_wealth_port - running_max_port) / running_max_port
    max_dd_port = drawdown_port.min()
    
    cumulative_wealth_bench = 1 + df_plot['Cum_Benchmark_Pct']
    running_max_bench = cumulative_wealth_bench.expanding().max()
    drawdown_bench = (cumulative_wealth_bench - running_max_bench) / running_max_bench
    max_dd_bench = drawdown_bench.min()
    
    # Sharpe
    sharpe_port = (df_plot['Portfolio_Return'].mean()) / vol_port if vol_port > 0 else 0
    ann_sharpe_port = sharpe_port * np.sqrt(periods_per_year)
    sharpe_bench = (df_plot['Benchmark_Return'].mean()) / vol_bench if vol_bench > 0 else 0
    ann_sharpe_bench = sharpe_bench * np.sqrt(periods_per_year)
    
    # Sortino
    downside_port = df_plot['Portfolio_Return'][df_plot['Portfolio_Return'] < 0]
    sortino_port = (df_plot['Portfolio_Return'].mean() / downside_port.std() * np.sqrt(periods_per_year)) if len(downside_port) > 0 else np.inf
    
    # Calmar
    calmar_port = ann_return_port / abs(max_dd_port) if max_dd_port != 0 else np.inf
    
    # Info Ratio
    excess_returns = df_plot['Portfolio_Return'] - df_plot['Benchmark_Return']
    info_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else np.inf
    
    # Win Rate
    win_rate = (df_plot['Portfolio_Return'] > df_plot['Benchmark_Return']).sum() / len(df_plot)
    
    return {
        'total_return_port': total_return_port,
        'total_return_bench': total_return_bench,
        'ann_return_port': ann_return_port,
        'ann_return_bench': ann_return_bench,
        'ann_vol_port': ann_vol_port,
        'ann_vol_bench': ann_vol_bench,
        'max_dd_port': max_dd_port,
        'max_dd_bench': max_dd_bench,
        'ann_sharpe_port': ann_sharpe_port,
        'ann_sharpe_bench': ann_sharpe_bench,
        'sortino_port': sortino_port,
        'calmar_port': calmar_port,
        'info_ratio': info_ratio,
        'win_rate': win_rate
    }


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def mostrar_resultados_comparativos(resultados_todos):
    """
    Muestra resultados comparativos de los modelos ejecutados.
    
    Si hay múltiples modelos (2 o 3), primero muestra panel comparativo.
    Luego muestra detalles individuales para cada modelo.
    """
    
    # ========================================================================
    # TABLA COMPARATIVA (solo si hay 2+ modelos)
    # ========================================================================
    if len(resultados_todos) > 1:
        st.markdown("# 📊 Comparación entre Modelos")
        st.markdown("### Tabla Comparativa de Métricas")
        tabla_comp = generar_tabla_comparativa(resultados_todos)
        st.dataframe(tabla_comp, use_container_width=True)
        st.markdown("---")
    
    # ========================================================================
    # GRÁFICOS INDIVIDUALES
    # ========================================================================
    for resultado_dict in resultados_todos:
        nombre = resultado_dict['nombre']
        resultado = resultado_dict['resultado']
        
        st.markdown(f"## 📊 {nombre}")
        _mostrar_detalle_modelo(nombre, resultado)
        st.markdown("---")


def _mostrar_detalle_modelo(nombre, resultado):
    """Muestra el detalle completo de un modelo individual."""
    
    # Generar gráfico de 6 paneles
    fig = generar_grafico_general(resultado['df_plot'], nombre, resultado['metricas'])
    st.pyplot(fig)
    
    # Tabla de métricas en 3 columnas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Retorno Total", f"{resultado['metricas']['total_return_port']*100:.2f}%")
        st.metric("Retorno Anualizado", f"{resultado['metricas']['ann_return_port']*100:.2f}%")
    with col2:
        st.metric("Volatilidad Anual", f"{resultado['metricas']['ann_vol_port']*100:.2f}%")
        st.metric("Sharpe Ratio", f"{resultado['metricas']['ann_sharpe_port']:.3f}")
    with col3:
        st.metric("Max Drawdown", f"{resultado['metricas']['max_dd_port']*100:.2f}%")
        st.metric("Win Rate", f"{resultado['metricas']['win_rate']*100:.1f}%")
    
    # Tabla de acciones seleccionadas por período
    st.markdown("---")
    st.markdown("### 📋 Acciones Seleccionadas por Período de Rebalanceo")
    
    df_weights = generar_tabla_acciones(resultado['portfolio_weights'])
    st.dataframe(df_weights, use_container_width=True, height=400)


def generar_tabla_acciones(portfolio_weights):
    """
    Genera tabla con las acciones seleccionadas en cada período.
    
    Args:
        portfolio_weights: Lista de dicts con Date y pesos por acción
        
    Returns:
        pd.DataFrame con fechas en filas y acciones con sus pesos
    """
    if not portfolio_weights:
        return pd.DataFrame()
    
    # Convertir a DataFrame
    df_weights = pd.DataFrame(portfolio_weights)
    
    # Fecha como índice
    df_weights['Date'] = pd.to_datetime(df_weights['Date'])
    df_weights = df_weights.set_index('Date')
    
    # Ordenar por fecha
    df_weights = df_weights.sort_index()
    
    # Formatear: mostrar solo acciones con peso > 0, ordenadas por peso
    df_display = pd.DataFrame()
    
    for fecha in df_weights.index:
        row = df_weights.loc[fecha]
        # Filtrar acciones con peso > 0
        activos = {ticker: peso for ticker, peso in row.items() if peso > 1e-6}
        # Ordenar por peso descendente
        activos_ordenados = dict(sorted(activos.items(), key=lambda x: x[1], reverse=True))
        
        # Crear string con formato "TICKER (XX.X%)"
        acciones_str = ", ".join([f"{ticker} ({peso*100:.1f}%)" for ticker, peso in activos_ordenados.items()])
        num_acciones = len(activos_ordenados)
        
        df_display = pd.concat([df_display, pd.DataFrame({
            'Fecha': [fecha.strftime('%Y-%m-%d')],
            '# Acciones': [num_acciones],
            'Acciones Seleccionadas (Peso)': [acciones_str]
        })], ignore_index=True)
    
    return df_display


def generar_tabla_comparativa(resultados_todos):
    """
    Tabla comparativa con métricas de todos los modelos en columnas.
    
    Args:
        resultados_todos: Lista de dicts con keys: 'nombre', 'resultado'
    
    Returns:
        pd.DataFrame con métricas en filas, modelos en columnas
    """
    
    data = {}
    
    # Recolectar métricas de cada modelo
    for resultado_dict in resultados_todos:
        nombre = resultado_dict['nombre']
        metricas = resultado_dict['resultado']['metricas']
        
        data[nombre] = {
            'Retorno Total (%)': f"{metricas['total_return_port']*100:.2f}",
            'Retorno Anualizado (%)': f"{metricas['ann_return_port']*100:.2f}",
            'Volatilidad Anual (%)': f"{metricas['ann_vol_port']*100:.2f}",
            'Sharpe Ratio': f"{metricas['ann_sharpe_port']:.3f}",
            'Max Drawdown (%)': f"{metricas['max_dd_port']*100:.2f}",
            'Win Rate (%)': f"{metricas['win_rate']*100:.1f}"
        }
    
    # Agregar columna de Benchmark
    metricas_bench = resultados_todos[0]['resultado']['metricas']  # Todos tienen mismo benchmark
    data['Benchmark'] = {
        'Retorno Total (%)': f"{metricas_bench['total_return_bench']*100:.2f}",
        'Retorno Anualizado (%)': f"{metricas_bench['ann_return_bench']*100:.2f}",
        'Volatilidad Anual (%)': f"{metricas_bench['ann_vol_bench']*100:.2f}",
        'Sharpe Ratio': f"{metricas_bench['ann_sharpe_bench']:.3f}",
        'Max Drawdown (%)': f"{metricas_bench['max_dd_bench']*100:.2f}",
        'Win Rate (%)': "N/A"  # Benchmark no tiene win rate
    }
    
    # Crear DataFrame
    df_comp = pd.DataFrame(data)
    
    return df_comp


def generar_grafico_general(df_plot, titulo_modelo, metricas):
    """
    Genera el gráfico de 6 paneles.
    """
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Retornos Acumulados
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_plot['Date'], df_plot['Cum_Portfolio_Pct'] * 100, 
             label='Portfolio Optimizado', linewidth=2.5, color='#2E86AB')
    ax1.plot(df_plot['Date'], df_plot['Cum_Benchmark_Pct'] * 100, 
             label='Benchmark (SPY)', linewidth=2.5, color='#A23B72', linestyle='--')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.fill_between(df_plot['Date'], df_plot['Cum_Portfolio_Pct'] * 100, 
                      df_plot['Cum_Benchmark_Pct'] * 100, 
                      where=(df_plot['Cum_Portfolio_Pct'] >= df_plot['Cum_Benchmark_Pct']),
                      alpha=0.3, color='green', label='Outperformance')
    ax1.fill_between(df_plot['Date'], df_plot['Cum_Portfolio_Pct'] * 100, 
                      df_plot['Cum_Benchmark_Pct'] * 100, 
                      where=(df_plot['Cum_Portfolio_Pct'] < df_plot['Cum_Benchmark_Pct']),
                      alpha=0.3, color='red', label='Underperformance')
    ax1.set_ylabel('Retorno Acumulado (%)', fontsize=11)
    ax1.set_title(f'Desempeño del Portfolio - {titulo_modelo}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    cumulative_wealth_port = 1 + df_plot['Cum_Portfolio_Pct']
    running_max_port = cumulative_wealth_port.expanding().max()
    drawdown_port = (cumulative_wealth_port - running_max_port) / running_max_port * 100
    
    cumulative_wealth_bench = 1 + df_plot['Cum_Benchmark_Pct']
    running_max_bench = cumulative_wealth_bench.expanding().max()
    drawdown_bench = (cumulative_wealth_bench - running_max_bench) / running_max_bench * 100
    
    ax2.fill_between(df_plot['Date'], 0, drawdown_port, alpha=0.5, color='#2E86AB', label='Portfolio')
    ax2.fill_between(df_plot['Date'], 0, drawdown_bench, alpha=0.3, color='#A23B72', label='Benchmark')
    ax2.axhline(y=metricas['max_dd_port'] * 100, color='red', linestyle='--', linewidth=1, 
                label=f'Max DD Port: {metricas["max_dd_port"]*100:.2f}%')
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_xlabel('Fecha', fontsize=11)
    ax2.set_title('Drawdown (Caída desde Máximo)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Retornos por Período
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(df_plot['Date'], df_plot['Portfolio_Return'] * 100, 
            alpha=0.7, color='#2E86AB', label='Portfolio', width=20)
    ax3.bar(df_plot['Date'], df_plot['Benchmark_Return'] * 100, 
            alpha=0.5, color='#A23B72', label='Benchmark', width=20)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Retorno por Período (%)', fontsize=11)
    ax3.set_xlabel('Fecha', fontsize=11)
    ax3.set_title('Retornos Periódicos', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Distribución de Retornos
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(df_plot['Portfolio_Return'] * 100, bins=15, alpha=0.7, 
             color='#2E86AB', label='Portfolio', edgecolor='black')
    ax4.hist(df_plot['Benchmark_Return'] * 100, bins=15, alpha=0.5, 
             color='#A23B72', label='Benchmark', edgecolor='black')
    ax4.axvline(df_plot['Portfolio_Return'].mean() * 100, color='#2E86AB', 
                linestyle='--', linewidth=2, label='Media Port')
    ax4.axvline(df_plot['Benchmark_Return'].mean() * 100, color='#A23B72', 
                linestyle='--', linewidth=2, label='Media Bench')
    ax4.set_xlabel('Retorno (%)', fontsize=11)
    ax4.set_ylabel('Frecuencia', fontsize=11)
    ax4.set_title('Distribución de Retornos', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Tabla de métricas
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    metrics_data = [
        ['MÉTRICA', 'VALOR'],
        ['─' * 25, '─' * 12],
        ['Retorno Total', f"{metricas['total_return_port']*100:.2f}%"],
        ['Retorno Anualizado', f"{metricas['ann_return_port']*100:.2f}%"],
        ['Volatilidad Anual', f"{metricas['ann_vol_port']*100:.2f}%"],
        ['Sharpe Ratio', f"{metricas['ann_sharpe_port']:.3f}"],
        ['Sortino Ratio', f"{metricas['sortino_port']:.3f}"],
        ['Max Drawdown', f"{metricas['max_dd_port']*100:.2f}%"],
        ['Calmar Ratio', f"{metricas['calmar_port']:.3f}"],
        ['Info Ratio', f"{metricas['info_ratio']:.3f}"],
        ['Win Rate', f"{metricas['win_rate']*100:.1f}%"],
    ]
    
    table = ax5.table(cellText=metrics_data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(metrics_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#2E86AB')
                cell.set_text_props(weight='bold', color='white')
            elif i == 1:
                cell.set_facecolor('#E8E8E8')
            else:
                cell.set_facecolor('#E3F2FD')
    
    plt.suptitle(f'Análisis Completo - {titulo_modelo}', fontsize=15, fontweight='bold', y=0.995)
    
    return fig
