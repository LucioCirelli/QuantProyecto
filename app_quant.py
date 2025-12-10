"""
APP_QUANT.PY - Aplicación Streamlit

TAB 1: PREPROCESAMIENTO
- Cargar datos (S&P 500)
- Configurar parámetros de modelos

TAB 2: BACKTESTING & RESULTADOS
- Seleccionar período de backtesting
- Seleccionar modelos a comparar (1, 2 o 3)
- Ejecutar backtesting con rebalanceo mensual
- Visualizar resultados

"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INICIALIZACIÓN DEL SESSION STATE
# ============================================================================

if 'datos_cargados' not in st.session_state:
    st.session_state.datos_cargados = False
    st.session_state.df_tickers = None
    st.session_state.df_spy = None
    st.session_state.preprocess_start_year = 2010
    st.session_state.preprocess_end_year = 2020
    st.session_state.backtest_start_year = 2021
    st.session_state.backtest_end_year = 2023
    st.session_state.window_meses = 100
    # Las fechas se calculan automáticamente al cargar datos
    st.session_state.backtest_inicio = pd.to_datetime('2021-01-01')
    st.session_state.backtest_fin = min(pd.to_datetime('2023-12-31'), pd.Timestamp.now())

# Parámetros de modelos (defaults)
if 'parametros_min_riesgo' not in st.session_state:
    st.session_state.parametros_min_riesgo = {
        'max_acciones': 10,
        'w_minimo': 0.05,
        'w_maximo': 0.3,
        'rendimiento_minimo': np.log(1.015)
    }

if 'parametros_max_beneficio' not in st.session_state:
    st.session_state.parametros_max_beneficio = {
        'max_acciones': 10,
        'w_minimo': 0.05,
        'w_maximo': 0.3,
        'rendimiento_minimo': np.log(1.015)
    }

if 'parametros_mvo' not in st.session_state:
    st.session_state.parametros_mvo = {
        'aversion_riesgo': 2.0,
        'max_activos': 30,
        'min_activos': 20,
        'peso_maximo': 0.10,
        'peso_minimo': 0.015,
        'turnover_limit': 0.75
    }

if 'parametros_preprocesamiento' not in st.session_state:
    st.session_state.parametros_preprocesamiento = {
        'peso_media': 0.5,
        'peso_momentum': 0.5,
        'meses_momentum': 3,
        'nivel_confianza': 1.96
    }

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("📊 Portfolio Optimizer")
    st.markdown("---")
    st.markdown("""
    1. 📥 **Preprocesamiento**: Carga datos
    2. 🧪 **Backtesting**: Ejecuta y compara modelos
    
    **Modelos disponibles:**
    - Minimizador de Riesgo
    - Maximizador de Beneficio
    - Robusto
    """)
    
    st.markdown("---")
    
    if st.session_state.datos_cargados:
        st.success("✅ Datos cargados")
        st.metric("Tickers", len(st.session_state.df_tickers['Ticker'].unique()))
        st.metric("Observaciones", len(st.session_state.df_tickers))
    else:
        st.warning("⚠️ Cargar datos en Preprocesamiento")

# ============================================================================
# TABS PRINCIPALES
# ============================================================================

tab1, tab2 = st.tabs(["📥 Preprocesamiento", "🧪 Backtesting & Resultados"])

# ============================================================================
# TAB 1: PREPROCESAMIENTO
# ============================================================================

with tab1:
    st.header("📥 Preprocesamiento de Datos")
    
    st.markdown("""
    Este tab carga los datos del S&P 500 y configura los parámetros de cada modelo.
    
    Los datos descargados cubren ambos períodos (preprocesamiento y backtesting) con frecuencia mensual.
    """)
    
    st.markdown("---")
    
    # ========================================
    # SECCIÓN: CARGA DE DATOS
    # ========================================
    st.subheader("1️⃣ Carga de Datos")
    
    st.markdown("**📚 Periodo de Preprocesamiento**")
    col1, col2 = st.columns(2)
    with col1:
        preprocess_start_year = st.number_input(
            "Año inicial preprocesamiento",
            min_value=1990,
            max_value=2025,
            value=st.session_state.preprocess_start_year,
            step=1,
            help="Año de inicio para entrenar el modelo"
        )
    with col2:
        preprocess_end_year = st.number_input(
            "Año final preprocesamiento",
            min_value=2000,
            max_value=2025,
            value=st.session_state.preprocess_end_year,
            step=1,
            help="Año final para entrenar el modelo"
        )
    
    st.markdown("---")
    
    st.markdown("**🧪 Periodo de Backtesting**")
    col3, col4 = st.columns(2)
    with col3:
        backtest_start_year = st.number_input(
            "Año inicial backtesting",
            min_value=2000,
            max_value=2025,
            value=st.session_state.backtest_start_year,
            step=1,
            help="Año de inicio del backtesting (debe ser > año final preprocesamiento)"
        )
    with col4:
        backtest_end_year = st.number_input(
            "Año final backtesting",
            min_value=2000,
            max_value=2025,
            value=st.session_state.backtest_end_year,
            step=1,
            help="Año final del backtesting"
        )
    
    # Validación: backtest debe ser después del preprocess
    if backtest_start_year <= preprocess_end_year:
        st.warning(f"⚠️ El año inicial de backtesting ({backtest_start_year}) debe ser posterior al año final de preprocesamiento ({preprocess_end_year})")
    
    st.markdown("---")
    
    st.markdown("**⚙️ Configuración de Backtesting:**")
    st.info("ℹ️ Las fechas se calculan automáticamente: inicio = 1 de enero del año inicial, fin = 31 de diciembre del año final (o fecha actual si no se alcanzó)")
    
    window_meses = st.number_input(
        "Ventana de entrenamiento (meses)",
        min_value=24,
        max_value=120,
        value=st.session_state.get('window_meses', 100),
        step=12,
        help="Meses históricos para calcular inputs en cada rebalanceo"
    )
    
    if st.button("🔄 Cargar Datos", type="primary", use_container_width=True):
        # Validación antes de cargar
        if backtest_start_year <= preprocess_end_year:
            st.error(f"❌ Error: El año inicial de backtesting ({backtest_start_year}) debe ser posterior al año final de preprocesamiento ({preprocess_end_year})")
        else:
            with st.spinner("Descargando datos del S&P 500 y SPY..."):
                try:
                    # Importar función de carga
                    from utils_backend.utils.CargarDatos import descargar_sp500_mensual, descargar_spy
                    
                    # Descargar datos desde el inicio del preprocesamiento hasta el fin del backtesting
                    # para tener todos los datos necesarios
                    overall_start = preprocess_start_year
                    overall_end = backtest_end_year
                    
                    df_tickers_crudo = descargar_sp500_mensual(overall_start, overall_end, guardar_csv=False)
                    df_spy = descargar_spy(overall_start, overall_end)
                    
                    # Procesar datos
                    df_tickers_crudo['Date'] = pd.to_datetime(df_tickers_crudo['Date'])
                    df_tickers_crudo = df_tickers_crudo.sort_values(['Ticker', 'Date'])
                    df_tickers_crudo['Return'] = df_tickers_crudo.groupby('Ticker')['Close'].transform(
                        lambda x: np.log(x / x.shift(1))
                    )
                    df_tickers_crudo = df_tickers_crudo.dropna(subset=['Return'])
                    
                    # Procesar SPY
                    df_spy['Date'] = pd.to_datetime(df_spy['Date'])
                    df_spy = df_spy.sort_values(['Ticker', 'Date'])
                    df_spy['Return'] = df_spy.groupby('Ticker')['Close'].transform(
                        lambda x: np.log(x / x.shift(1))
                    )
                    df_spy = df_spy.dropna(subset=['Return'])
                    
                    # Calcular fechas de backtesting automáticamente
                    # Inicio: 1 de enero del año inicial de backtesting
                    backtest_inicio_calc = pd.to_datetime(f"{backtest_start_year}-01-01")
                    
                    # Fin: 31 de diciembre del año final de backtesting, o fecha actual si no se alcanzó
                    end_of_year = pd.to_datetime(f"{backtest_end_year}-12-31")
                    today = pd.Timestamp.now()
                    backtest_fin_calc = min(end_of_year, today)
                    
                    # Guardar en session state
                    st.session_state.df_tickers = df_tickers_crudo
                    st.session_state.df_spy = df_spy
                    st.session_state.datos_cargados = True
                    st.session_state.preprocess_start_year = preprocess_start_year
                    st.session_state.preprocess_end_year = preprocess_end_year
                    st.session_state.backtest_start_year = backtest_start_year
                    st.session_state.backtest_end_year = backtest_end_year
                    st.session_state.window_meses = window_meses
                    st.session_state.backtest_inicio = backtest_inicio_calc
                    st.session_state.backtest_fin = backtest_fin_calc
                    
                    st.success(f"✅ Datos cargados: {len(df_tickers_crudo['Ticker'].unique())} tickers, "
                              f"{len(df_tickers_crudo)} observaciones")
                    st.success(f"📚 Preprocesamiento: {preprocess_start_year}-{preprocess_end_year} | 🧪 Backtesting: {backtest_start_year}-{backtest_end_year}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error al cargar datos: {str(e)}")
                    st.exception(e)
    
    if st.session_state.datos_cargados:
        st.info(f"📚 Preprocesamiento: {st.session_state.preprocess_start_year}-{st.session_state.preprocess_end_year} | 🧪 Backtesting: {st.session_state.backtest_start_year}-{st.session_state.backtest_end_year}")
        
        # Preview de datos
        with st.expander("👁️ Ver preview de datos"):
            st.dataframe(st.session_state.df_tickers, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # SECCIÓN: PARÁMETROS DE PREPROCESAMIENTO
    # ========================================
    st.subheader("2️⃣ Parámetros de Preprocesamiento")
    st.caption("Estos parámetros aplican al cálculo de inputs (μ, Σ, δ) para todos los modelos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        peso_media = st.slider(
            "Peso media histórica",
            0.0, 1.0,
            st.session_state.parametros_preprocesamiento['peso_media'],
            step=0.1,
            help="Peso de la media histórica en μ (el resto es momentum)",
            key='peso_media'
        )
    
    with col2:
        meses_momentum = st.number_input(
            "Meses para momentum",
            1, 12,
            st.session_state.parametros_preprocesamiento['meses_momentum'],
            help="Últimos N meses para calcular momentum",
            key='meses_momentum'
        )
    
    with col3:
        nivel_confianza = st.number_input(
            "Nivel confianza (δ)",
            1.0, 3.0,
            st.session_state.parametros_preprocesamiento['nivel_confianza'],
            step=0.1,
            help="Z-score para calcular δ = z * σ / √T (1.96 = 95%)",
            key='nivel_confianza'
        )
    
    with col4:
        if st.button("💾 Guardar Preprocesamiento", use_container_width=True):
            st.session_state.parametros_preprocesamiento = {
                'peso_media': peso_media,
                'peso_momentum': 1.0 - peso_media,
                'meses_momentum': meses_momentum,
                'nivel_confianza': nivel_confianza
            }
            st.success("✅ Guardado")
    
    with st.expander("ℹ️ Fórmulas de preprocesamiento"):
        st.latex(r"\mu = w_{media} \cdot \bar{r} + w_{momentum} \cdot r_{momentum}")
        st.latex(r"\Sigma = Cov(r_t)")
        st.latex(r"\delta = z \cdot \frac{\sigma}{\sqrt{T}}")
    
    st.markdown("---")
    
    # ========================================
    # SECCIÓN: PARÁMETROS DE MODELOS
    # ========================================
    st.subheader("3️⃣ Parámetros de Modelos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🎲 Modelo Minimizador de Riesgo")
        max_acc_est = st.number_input("Max acciones", 2, 100, 
                                      st.session_state.parametros_min_riesgo['max_acciones'],
                                      key='max_est')
        w_min_est = st.number_input("Peso mínimo", 0.01, 0.2, 
                                    st.session_state.parametros_min_riesgo['w_minimo'],
                                    key='wmin_est')
        w_max_est = st.number_input("Peso máximo", 0.1, 0.7, 
                                    st.session_state.parametros_min_riesgo['w_maximo'],
                                    key='wmax_est')
        
        if st.button("💾 Guardar", key='save_est', use_container_width=True):
            st.session_state.parametros_min_riesgo = {
                'max_acciones': max_acc_est,
                'w_minimo': w_min_est,
                'w_maximo': w_max_est,
                'rendimiento_minimo': np.log(1.015)
            }
            st.success("✅ Guardado")
    
    with col2:
        st.markdown("##### 🛡️ Modelo Maximizador de Beneficio")
        max_acc_rob = st.number_input("Max acciones", 2, 100, 
                                      st.session_state.parametros_max_beneficio['max_acciones'],
                                      key='max_rob')
        w_min_rob = st.number_input("Peso mínimo", 0.01, 0.2, 
                                    st.session_state.parametros_max_beneficio['w_minimo'],
                                    key='wmin_rob')
        w_max_rob = st.number_input("Peso máximo", 0.1, 0.7, 
                                    st.session_state.parametros_max_beneficio['w_maximo'],
                                    key='wmax_rob')
        
        if st.button("💾 Guardar", key='save_rob', use_container_width=True):
            st.session_state.parametros_max_beneficio = {
                'max_acciones': max_acc_rob,
                'w_minimo': w_min_rob,
                'w_maximo': w_max_rob,
                'rendimiento_minimo': np.log(1.015)
            }
            st.success("✅ Guardado")
    
    with col3:
        st.markdown("##### 🎯 Robusto")
        
        aversion = st.number_input(
            "Aversión al riesgo",
            0.5, 10.0,
            st.session_state.parametros_mvo['aversion_riesgo'],
            step=0.5,
            key='aversion_mvo'
        )
        
        max_act_mvo = st.number_input(
            "Max activos",
            10, 100,
            st.session_state.parametros_mvo['max_activos'],
            key='max_mvo'
        )
        
        min_act_mvo = st.number_input(
            "Min activos",
            2, 30,
            st.session_state.parametros_mvo['min_activos'],
            key='min_mvo'
        )
        
        w_max_mvo = st.number_input(
            "Peso máximo",
            0.05, 0.70,
            st.session_state.parametros_mvo['peso_maximo'],
            step=0.01,
            format="%.2f",
            key='wmax_mvo'
        )
        
        w_min_mvo = st.number_input(
            "Peso mínimo",
            0.001, 0.10,
            st.session_state.parametros_mvo['peso_minimo'],
            step=0.001,
            format="%.3f",
            key='wmin_mvo'
        )
        
        if st.button("💾 Guardar", key='save_mvo', use_container_width=True):
            st.session_state.parametros_mvo = {
                'aversion_riesgo': aversion,
                'max_activos': max_act_mvo,
                'min_activos': min_act_mvo,
                'peso_maximo': w_max_mvo,
                'peso_minimo': w_min_mvo
            }
            st.success("✅ Guardado")

# ============================================================================
# TAB 2: BACKTESTING & RESULTADOS
# ============================================================================

with tab2:
    st.header("🧪 Backtesting & Resultados")
    
    if not st.session_state.datos_cargados:
        st.warning("⚠️ Primero debes cargar los datos en la pestaña **Preprocesamiento**")
        st.stop()
    
    st.markdown("""
    Ejecuta backtesting con rebalanceo mensual y compara los modelos seleccionados.
    
    **Periodo de evaluación:** Los modelos se evalúan en el periodo de backtesting configurado.
    """)
    
    st.markdown("---")
    
    # ========================================
    # CONFIGURACIÓN DE BACKTESTING
    # ========================================
    st.subheader("1️⃣ Configuración del Backtesting")
    
    # Mostrar parámetros configurados en Tab 1
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ventana entrenamiento", f"{st.session_state.window_meses} meses")
    with col2:
        inicio_str = st.session_state.backtest_inicio.strftime('%Y-%m-%d')
        st.metric("Inicio backtesting", inicio_str)
    with col3:
        fin_str = st.session_state.backtest_fin.strftime('%Y-%m-%d')
        st.metric("Fin backtesting", fin_str)
    
    st.info("💡 Para cambiar estos parámetros, ve al Tab 1 - Preprocesamiento")
    
    # Obtener valores de session state
    start_backtest = pd.to_datetime(st.session_state.backtest_inicio)
    end_backtest = pd.to_datetime(st.session_state.backtest_fin)
    window_meses = st.session_state.window_meses
    
    # Frecuencia de rebalanceo (mantener aquí porque es operacional)
    rebalance_freq = 1
    
    st.markdown("---")
    
    # ========================================
    # SELECCIÓN DE MODELOS
    # ========================================
    st.subheader("2️⃣ Selección de Modelos a Comparar")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_estocastico = st.checkbox("🎲 Modelo Minimizador de Riesgo", value=False)
    with col2:
        run_robust = st.checkbox("🛡️ Modelo Maximizador de Beneficio", value=False)
    with col3:
        run_mvo = st.checkbox("🎯 Robusto", value=False)
    
    modelos_seleccionados = []
    if run_estocastico:
        modelos_seleccionados.append(('minimizador_riesgo', 'Minimizador de Riesgo', st.session_state.parametros_min_riesgo))
    if run_robust:
        modelos_seleccionados.append(('maximizador_beneficio', 'Maximizador de Beneficio', st.session_state.parametros_max_beneficio))
    if run_mvo:
        modelos_seleccionados.append(('mvo', 'Robusto', st.session_state.parametros_mvo))
    
    if len(modelos_seleccionados) == 0:
        st.warning("⚠️ Debes seleccionar al menos un modelo")
        st.stop()
    
    st.info(f"✅ {len(modelos_seleccionados)} modelo(s) seleccionado(s)")
    
    st.markdown("---")
    
    # ========================================
    # EJECUTAR BACKTESTING
    # ========================================
    st.subheader("3️⃣ Ejecutar Backtesting")
    
    if st.button("🚀 EJECUTAR BACKTESTING", type="primary", use_container_width=True):
        
        # Importar funciones necesarias
        from utils_backend.funciones_backtesting import ejecutar_backtesting_completo
        
        # Contenedor para resultados
        resultados_todos = []
        
        # Ejecutar cada modelo
        for nombre_modelo, titulo_modelo, parametros in modelos_seleccionados:
            st.markdown(f"### Ejecutando: {titulo_modelo}")
            
            with st.spinner(f"Ejecutando backtesting para {titulo_modelo}..."):
                try:
                    resultado = ejecutar_backtesting_completo(
                        df_tickers=st.session_state.df_tickers,
                        df_spy=st.session_state.df_spy,
                        nombre_modelo=nombre_modelo,
                        parametros=parametros,
                        start_date=str(start_backtest),
                        end_date=str(end_backtest),
                        window_meses=window_meses,
                        rebalance_freq=rebalance_freq,
                        parametros_preprocesamiento=st.session_state.parametros_preprocesamiento
                    )
                    
                    resultados_todos.append({
                        'nombre': titulo_modelo,
                        'resultado': resultado
                    })
                    
                    st.success(f"✅ {titulo_modelo} completado")
                    
                except Exception as e:
                    st.error(f"❌ Error en {titulo_modelo}: {str(e)}")
                    st.exception(e)
        
        # Guardar resultados en session state
        if len(resultados_todos) > 0:
            st.session_state.resultados_backtesting = resultados_todos
            st.session_state.backtesting_ejecutado = True
            
            st.success("🎉 Backtesting completado para todos los modelos!")
            st.rerun()
    
    st.markdown("---")
    
    # ========================================
    # MOSTRAR RESULTADOS
    # ========================================
    if hasattr(st.session_state, 'backtesting_ejecutado') and st.session_state.backtesting_ejecutado:
        st.subheader("4️⃣ Resultados del Backtesting")
        
        # Importar función de visualización
        from utils_backend.funciones_backtesting import mostrar_resultados_comparativos
        
        # Mostrar gráficos comparativos
        mostrar_resultados_comparativos(st.session_state.resultados_backtesting)
        
    else:
        st.info("👆 Configura los parámetros y presiona el botón para ejecutar el backtesting")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("Portfolio Optimizer - Backtesting con Rebalanceo Mensual")
