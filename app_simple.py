"""
APP_SIMPLE.PY - Aplicación Streamlit simplificada de 2 tabs

TAB 1: PREPROCESAMIENTO
- Cargar datos (S&P 500)
- Configurar parámetros de modelos

TAB 2: BACKTESTING & RESULTADOS
- Seleccionar período de backtesting
- Seleccionar modelos a comparar (1, 2 o 3)
- Ejecutar backtesting con rebalanceo mensual (IGUAL QUE FRANCO)
- Mostrar gráficos IDÉNTICOS a DinamicMVO.ipynb

TODO ES CAJA NEGRA: Usuario clickea y ve resultados.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Portfolio Optimizer - Backtesting",
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
    st.session_state.start_year = 2010
    st.session_state.end_year = 2023

# Parámetros de modelos (defaults)
if 'parametros_estocastico' not in st.session_state:
    st.session_state.parametros_estocastico = {
        'max_acciones': 10,
        'w_minimo': 0.05,
        'w_maximo': 0.3,
        'rendimiento_minimo': np.log(1.015)
    }

if 'parametros_robust' not in st.session_state:
    st.session_state.parametros_robust = {
        'max_acciones': 10,
        'w_minimo': 0.05,
        'w_maximo': 0.3,
        'rendimiento_minimo': np.log(1.015)
    }

if 'parametros_franco' not in st.session_state:
    st.session_state.parametros_franco = {
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
    st.markdown("### 🎯 Aplicación Simplificada")
    st.markdown("""
    **2 Tabs:**
    1. 📥 **Preprocesamiento**: Carga datos
    2. 🧪 **Backtesting**: Ejecuta y compara modelos
    
    **Modelos disponibles:**
    - Estocástico (CVaR)
    - Robust Optimization
    - Franco (MVO Dinámico)
    """)
    
    st.markdown("---")
    
    if st.session_state.datos_cargados:
        st.success("✅ Datos cargados")
        st.metric("Tickers", len(st.session_state.df_tickers['Ticker'].unique()))
        st.metric("Observaciones", len(st.session_state.df_tickers))
    else:
        st.warning("⚠️ Cargar datos en Tab 1")

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
    """)
    
    st.markdown("---")
    
    # ========================================
    # SECCIÓN: CARGA DE DATOS
    # ========================================
    st.subheader("1️⃣ Carga de Datos")
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(
            "Año inicial",
            min_value=2000,
            max_value=2025,
            value=st.session_state.start_year,
            step=1
        )
    with col2:
        end_year = st.number_input(
            "Año final",
            min_value=2000,
            max_value=2025,
            value=st.session_state.end_year,
            step=1
        )
    
    if st.button("🔄 Cargar Datos", type="primary", use_container_width=True):
        with st.spinner("Descargando datos del S&P 500 y SPY..."):
            try:
                # Importar función de carga
                from Pipeline_Modelos_Propios.utils.CargarDatos import descargar_sp500_mensual, descargar_spy
                
                # Descargar datos
                df_tickers_crudo = descargar_sp500_mensual(start_year, end_year, guardar_csv=False)
                df_spy = descargar_spy(start_year, end_year)
                
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
                
                # Guardar en session state
                st.session_state.df_tickers = df_tickers_crudo
                st.session_state.df_spy = df_spy
                st.session_state.datos_cargados = True
                st.session_state.start_year = start_year
                st.session_state.end_year = end_year
                
                st.success(f"✅ Datos cargados: {len(df_tickers_crudo['Ticker'].unique())} tickers, "
                          f"{len(df_tickers_crudo)} observaciones")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error al cargar datos: {str(e)}")
                st.exception(e)
    
    if st.session_state.datos_cargados:
        st.info(f"📊 Datos cargados: {st.session_state.start_year} - {st.session_state.end_year}")
        
        # Preview de datos
        with st.expander("👁️ Ver preview de datos"):
            st.dataframe(st.session_state.df_tickers.head(20), use_container_width=True)
    
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
        st.markdown("##### 🎲 Modelo Estocástico")
        max_acc_est = st.number_input("Max acciones", 5, 30, 
                                      st.session_state.parametros_estocastico['max_acciones'],
                                      key='max_est')
        w_min_est = st.number_input("Peso mínimo", 0.01, 0.2, 
                                    st.session_state.parametros_estocastico['w_minimo'],
                                    key='wmin_est')
        w_max_est = st.number_input("Peso máximo", 0.1, 0.5, 
                                    st.session_state.parametros_estocastico['w_maximo'],
                                    key='wmax_est')
        
        if st.button("💾 Guardar", key='save_est', use_container_width=True):
            st.session_state.parametros_estocastico = {
                'max_acciones': max_acc_est,
                'w_minimo': w_min_est,
                'w_maximo': w_max_est,
                'rendimiento_minimo': np.log(1.015)
            }
            st.success("✅ Guardado")
    
    with col2:
        st.markdown("##### 🛡️ Robust Optimization")
        max_acc_rob = st.number_input("Max acciones", 5, 30, 
                                      st.session_state.parametros_robust['max_acciones'],
                                      key='max_rob')
        w_min_rob = st.number_input("Peso mínimo", 0.01, 0.2, 
                                    st.session_state.parametros_robust['w_minimo'],
                                    key='wmin_rob')
        w_max_rob = st.number_input("Peso máximo", 0.1, 0.5, 
                                    st.session_state.parametros_robust['w_maximo'],
                                    key='wmax_rob')
        
        if st.button("💾 Guardar", key='save_rob', use_container_width=True):
            st.session_state.parametros_robust = {
                'max_acciones': max_acc_rob,
                'w_minimo': w_min_rob,
                'w_maximo': w_max_rob,
                'rendimiento_minimo': np.log(1.015)
            }
            st.success("✅ Guardado")
    
    with col3:
        st.markdown("##### 🎯 Franco (MVO Dinámico)")
        
        aversion = st.number_input(
            "Aversión al riesgo",
            0.5, 10.0,
            st.session_state.parametros_franco['aversion_riesgo'],
            step=0.5,
            key='aversion_franco'
        )
        
        max_act_franco = st.number_input(
            "Max activos",
            10, 50,
            st.session_state.parametros_franco['max_activos'],
            key='max_franco'
        )
        
        min_act_franco = st.number_input(
            "Min activos",
            5, 30,
            st.session_state.parametros_franco['min_activos'],
            key='min_franco'
        )
        
        w_max_franco = st.number_input(
            "Peso máximo",
            0.05, 0.30,
            st.session_state.parametros_franco['peso_maximo'],
            step=0.01,
            format="%.2f",
            key='wmax_franco'
        )
        
        w_min_franco = st.number_input(
            "Peso mínimo",
            0.001, 0.10,
            st.session_state.parametros_franco['peso_minimo'],
            step=0.001,
            format="%.3f",
            key='wmin_franco'
        )
        
        turnover_franco = st.slider(
            "Turnover limit",
            0.0, 1.0,
            st.session_state.parametros_franco['turnover_limit'],
            step=0.05,
            help="Máximo cambio permitido en pesos (1.0 = sin restricción)",
            key='turnover_franco'
        )
        
        if st.button("💾 Guardar", key='save_franco', use_container_width=True):
            st.session_state.parametros_franco = {
                'aversion_riesgo': aversion,
                'max_activos': max_act_franco,
                'min_activos': min_act_franco,
                'peso_maximo': w_max_franco,
                'peso_minimo': w_min_franco,
                'turnover_limit': turnover_franco
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
    Ejecuta backtesting con rebalanceo mensual (método de Franco) y compara los modelos seleccionados.
    """)
    
    st.markdown("---")
    
    # ========================================
    # CONFIGURACIÓN DE BACKTESTING
    # ========================================
    st.subheader("1️⃣ Configuración del Backtesting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Obtener rango de fechas disponibles
        fechas_disponibles = sorted(st.session_state.df_tickers['Date'].unique())
        fecha_min = fechas_disponibles[0]
        fecha_max = fechas_disponibles[-1]
        
        # Calcular valor por defecto: último año de datos disponibles
        # Si hay más de 12 meses de datos, usar fecha_max - 12 meses, sino usar fecha_min
        from dateutil.relativedelta import relativedelta
        if (fecha_max - fecha_min).days > 365:
            default_start = fecha_max - relativedelta(months=12)
        else:
            default_start = fecha_min + relativedelta(months=1)
        
        # Asegurar que default_start esté dentro del rango
        if default_start < fecha_min:
            default_start = fecha_min
        if default_start > fecha_max:
            default_start = fecha_max
        
        start_backtest = st.date_input(
            "Fecha inicio backtesting",
            value=default_start,
            min_value=fecha_min,
            max_value=fecha_max,
            help="Fecha desde la cual iniciar el backtesting (debe tener suficiente historia previa)"
        )
    
    with col2:
        window_meses = st.number_input(
            "Ventana histórica (meses)",
            min_value=24,
            max_value=120,
            value=100,
            step=12,
            help="Cantidad de meses históricos para calcular inputs en cada rebalanceo"
        )
    
    with col3:
        rebalance_freq = st.selectbox(
            "Frecuencia rebalanceo",
            options=[1, 3, 6, 12],
            index=0,
            format_func=lambda x: f"{x} mes(es)",
            help="Cada cuántos meses rebalancear el portfolio"
        )
    
    st.markdown("---")
    
    # ========================================
    # SELECCIÓN DE MODELOS
    # ========================================
    st.subheader("2️⃣ Selección de Modelos a Comparar")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_estocastico = st.checkbox("🎲 Modelo Estocástico", value=False)
    with col2:
        run_robust = st.checkbox("🛡️ Robust Optimization", value=False)
    with col3:
        run_franco = st.checkbox("🎯 Franco (MVO Dinámico)", value=True)
    
    modelos_seleccionados = []
    if run_estocastico:
        modelos_seleccionados.append(('estocastico', 'Estocástico', st.session_state.parametros_estocastico))
    if run_robust:
        modelos_seleccionados.append(('robust', 'Robust', st.session_state.parametros_robust))
    if run_franco:
        modelos_seleccionados.append(('franco', 'Franco', st.session_state.parametros_franco))
    
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
        from Pipeline_Modelos_Propios.funciones_backtesting import ejecutar_backtesting_completo
        
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
        from Pipeline_Modelos_Propios.funciones_backtesting import mostrar_resultados_comparativos
        
        # Mostrar gráficos comparativos
        mostrar_resultados_comparativos(st.session_state.resultados_backtesting)
        
    else:
        st.info("👆 Configura los parámetros y presiona el botón para ejecutar el backtesting")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("Portfolio Optimizer - Backtesting con Rebalanceo Mensual | Adaptado de Franco DinamicMVO")
