# 📊 Portfolio Optimizer - Sistema de Backtesting Cuantitativo

Sistema avanzado de optimización y backtesting de portafolios de inversión implementando tres estrategias cuantitativas: **Minimizador de Riesgo**, **Maximizador de Beneficio** y **Mean-Variance Optimization (MVO) Robusto**.

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema completo de backtesting para estrategias de inversión cuantitativas con rebalanceo mensual. Permite comparar el desempeño de diferentes modelos de optimización de portafolio contra el benchmark S&P 500 (SPY).

### Características Principales

- ✅ **3 Modelos de Optimización**: Minimizador de Riesgo, Maximizador de Beneficio y MVO Robusto
- ✅ **Backtesting Dinámico**: Rebalanceo mensual con ventana móvil
- ✅ **Métricas Completas**: Sharpe, Sortino, Calmar, Information Ratio, Max Drawdown, Win Rate
- ✅ **Visualización Avanzada**: Gráficos de 6 paneles por modelo + comparativas
- ✅ **Tracking de Portafolio**: Tabla de acciones seleccionadas por período
- ✅ **Interfaz Web**: Aplicación Streamlit interactiva
- ✅ **Datos Automáticos**: Descarga de datos del S&P 500 vía `yfinance`

---

## 📁 Estructura del Proyecto

```
QuantProyecto/
│
├── app_quant.py                    # Aplicación Streamlit principal
├── README.md                       # Este archivo
│
└── utils_backend/                  # Módulos backend
    ├── funciones_backtesting.py    # Lógica de backtesting, métricas y visualización
    ├── ModeloMinimizadorRiesgo.py  # Modelo Pyomo: minimizar riesgo downside
    ├── ModeloMaximizadorBeneficio.py # Modelo Pyomo: maximizar beneficio ajustado
    ├── OptimizarCartera.py         # MVO estático (primer rebalanceo)
    ├── OptimizarCarteraDinamico.py # MVO dinámico con turnover
    ├── requirements.txt            # Dependencias Python
    │
    └── utils/                      # Utilidades auxiliares
        ├── CargarDatos.py          # Descarga datos de yfinance
        ├── Preprocesamiento.py     # Filtrado y limpieza de datos
        └── Postprocesamiento.py    # Exportación de resultados
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos

1. **Python 3.10+**
2. **Gurobi Optimizer** (licencia académica o comercial)
   - Descargar: [https://www.gurobi.com/downloads/](https://www.gurobi.com/downloads/)
   - Activar licencia: `grbgetkey YOUR-LICENSE-KEY`

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/LucioCirelli/QuantProyecto.git
cd QuantProyecto

# Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r utils_backend/requirements.txt
```

### Verificar Gurobi

```bash
python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

Debería mostrar la versión de Gurobi instalada (ej: `(11, 0, 0)`).

---

## 💻 Uso de la Aplicación

### Ejecutar la App

```bash
streamlit run app_quant.py
```

La aplicación se abrirá en `http://localhost:8501`

### Workflow Completo

#### **Tab 1: Preprocesamiento** 📥

1. **Configurar Períodos**:
   - **Preprocesamiento**: Período para entrenar modelos (ej: 2010-2020)
   - **Backtesting**: Período para evaluar estrategias (ej: 2021-2023)

2. **Configurar Parámetros de Preprocesamiento**:
   - `peso_media`: Peso de retorno promedio histórico (0-1)
   - `peso_momentum`: Peso de momentum reciente (0-1)
   - `meses_momentum`: Ventana de momentum (1-12 meses)
   - `nivel_confianza`: Z-score para intervalos (1.65, 1.96, 2.33)

3. **Configurar Modelos**:

   **Minimizador de Riesgo / Maximizador de Beneficio**:
   - `max_acciones`: Número máximo de activos (5-20)
   - `w_minimo`: Peso mínimo por acción (0.01-0.10)
   - `w_maximo`: Peso máximo por acción (0.10-0.50)
   - `rendimiento_minimo`: Retorno mínimo requerido (log scale)

   **Modelo Robusto (MVO)**:
   - `aversion_riesgo`: Aversión al riesgo λ (0.5-5.0)
   - `max_activos` / `min_activos`: Límites del portafolio
   - `peso_maximo` / `peso_minimo`: Límites de concentración
   - `turnover_limit`: Máximo rebalanceo permitido (0.3-1.0)

4. **Cargar Datos**: Click en botón "📥 Cargar Datos del S&P 500"

#### **Tab 2: Backtesting & Resultados** 🧪

1. **Seleccionar Período de Backtesting**:
   - Ajustar fechas de inicio/fin dentro del período configurado
   - Establecer ventana histórica (36-120 meses)
   - Definir frecuencia de rebalanceo (1-12 meses)

2. **Seleccionar Modelos**:
   - Elegir 1, 2 o 3 modelos para comparar
   - Opciones: Minimizador de Riesgo, Maximizador de Beneficio, Robusto

3. **Ejecutar Backtesting**: Click en "🚀 Ejecutar Backtesting"

4. **Visualizar Resultados**:
   - **Tabla Comparativa**: Métricas de todos los modelos vs. Benchmark
   - **Gráficos Individuales**: 6 paneles por modelo
     - Retornos acumulados
     - Drawdown
     - Retornos por período
     - Distribución de retornos
     - Tabla de métricas
   - **Tabla de Acciones**: Stocks seleccionados por período de rebalanceo

---

## 📊 Modelos Implementados

### 1. Minimizador de Riesgo

**Objetivo**: Minimizar la exposición al riesgo downside (CVaR, VaR, probabilidad de pérdida)

**Formulación Matemática**:
```
min   Σ(CVaR_i * w_i) + Σ(VaR_i * w_i) + Σ(P_loss_i * w_i)

s.t.  Σ(μ_i * w_i) ≥ r_min
      Σ w_i = 1
      w_min ≤ w_i ≤ w_max  ∀i ∈ Seleccionados
      w_i = 0               ∀i ∉ Seleccionados
      |Seleccionados| ≤ max_acciones
```

**Características**:
- Usa CVaR (95%), VaR (95%) y probabilidad de pérdida empírica
- Ideal para inversionistas conservadores
- Penaliza fuertemente el riesgo de cola

### 2. Maximizador de Beneficio

**Objetivo**: Maximizar retorno esperado ajustado por riesgo

**Formulación Matemática**:
```
max   Σ(μ_i * w_i) - λ * [Σ(CVaR_i * w_i) + Σ(VaR_i * w_i)]

s.t.  Σ(μ_i * w_i) ≥ r_min
      Σ w_i = 1
      w_min ≤ w_i ≤ w_max  ∀i ∈ Seleccionados
      w_i = 0               ∀i ∉ Seleccionados
      |Seleccionados| ≤ max_acciones
```

**Características**:
- Balance entre retorno y riesgo downside
- Más agresivo que Minimizador de Riesgo
- λ = 0.5 (ajustable en código)

### 3. Modelo Robusto (MVO)

**Objetivo**: Optimización media-varianza robusta con control de turnover

**Formulación Matemática**:
```
min   λ * (w^T Σ w) - w^T μ + δ^T |w|

s.t.  Σ w_i = 1
      min_activos ≤ |Seleccionados| ≤ max_activos
      w_min ≤ w_i ≤ w_max  ∀i
      ||w - w_prev||₁ ≤ turnover_limit  (rebalanceos posteriores)
```

**Características**:
- Usa parámetro δ (incertidumbre del retorno esperado)
- Control de turnover para reducir costos de transacción
- Versión estática (primer período) y dinámica (posteriores)

---

## 📈 Métricas Calculadas

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **Retorno Total** | $(V_T - V_0) / V_0$ | Ganancia/pérdida acumulada |
| **Retorno Anualizado** | $(1 + R_{total})^{1/T} - 1$ | Retorno promedio anual |
| **Volatilidad Anual** | $\sigma \sqrt{252/T}$ | Riesgo anualizado |
| **Sharpe Ratio** | $(R_p - R_f) / \sigma_p$ | Retorno ajustado por riesgo total |
| **Sortino Ratio** | $(R_p - R_f) / \sigma_{down}$ | Retorno ajustado por riesgo downside |
| **Max Drawdown** | $\max_t [(V_{max,t} - V_t) / V_{max,t}]$ | Peor caída desde máximo |
| **Calmar Ratio** | $R_{ann} / \|DD_{max}\|$ | Retorno por unidad de drawdown |
| **Information Ratio** | $(R_p - R_b) / \sigma_{excess}$ | Retorno activo ajustado por tracking error |
| **Win Rate** | $\#(R_p > R_b) / \#periodos$ | % de períodos superando benchmark |

---

## 🔬 Detalles Técnicos

### Cálculo de Inputs (Preprocesamiento)

Para cada período de rebalanceo $t$:

1. **Retorno Esperado** ($\mu$):
   $$\mu_i = w_{media} \cdot \bar{R}_i + w_{momentum} \cdot R_{i,momentum}$$

2. **Matriz de Covarianzas** ($\Sigma$):
   $$\Sigma = \text{Cov}(R_{window})$$

3. **Desviación Estándar** ($\sigma$):
   $$\sigma_i = \sqrt{\Sigma_{ii}}$$

4. **Incertidumbre del Retorno** ($\delta$):
   $$\delta_i = z_{\alpha} \cdot \frac{\sigma_i}{\sqrt{T}}$$

5. **CVaR** (Conditional Value at Risk):
   $$\text{CVaR}_{95\%,i} = |\mu_i + 2.06 \cdot \sigma_i|$$

6. **VaR** (Value at Risk):
   $$\text{VaR}_{95\%,i} = |\mu_i + 1.65 \cdot \sigma_i|$$

7. **Probabilidad de Pérdida**:
   $$P_{loss,i} = \frac{\#(R_i < 0)}{\#observaciones}$$

### Proceso de Backtesting

```
FOR cada fecha de rebalanceo t:
    1. Extraer ventana histórica [t - window_meses, t)
    2. Calcular inputs: μ, Σ, δ, σ, CVaR, VaR, P_loss
    3. Ejecutar modelo de optimización → w_opt
    4. Calcular retorno del período [t, t + rebalance_freq)
    5. Actualizar métricas acumuladas
END FOR

Generar gráficos y tablas comparativas
```

---

## 🛠️ Solución de Problemas

### Error: "Gurobi license not found"

**Solución**:
```bash
# Activar licencia académica
grbgetkey YOUR-LICENSE-KEY

# Verificar licencia
python -c "from gurobipy import *; m = Model()"
```

### Error: "ModuleNotFoundError: No module named 'pyomo'"

**Solución**:
```bash
pip install pyomo gurobipy
```

### Error: "Solver (gurobi) not found"

**Solución**:
```bash
# Instalar Gurobi desde https://www.gurobi.com/downloads/
pip install gurobipy
```

### Advertencia: "No data available for ticker"

**Causa**: Algunos tickers del S&P 500 pueden no tener datos históricos completos.

**Solución**: El sistema automáticamente filtra tickers con menos de 12 observaciones.

### Backtesting muy lento

**Optimizaciones**:
- Reducir `window_meses` (ej: 60 en vez de 100)
- Aumentar `rebalance_freq` (ej: 3 meses en vez de 1)
- Reducir período de backtesting

---

## 📚 Referencias

### Papers Académicos

1. **Markowitz, H. (1952)**. "Portfolio Selection". *Journal of Finance*, 7(1), 77-91.
2. **Rockafellar, R. T., & Uryasev, S. (2000)**. "Optimization of conditional value-at-risk". *Journal of Risk*, 2, 21-42.
3. **DeMiguel, V., Garlappi, L., & Uppal, R. (2009)**. "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?". *Review of Financial Studies*, 22(5), 1915-1953.

### Librerías Utilizadas

- **Pyomo**: Modelado de optimización algebraica
- **Gurobi**: Solver de optimización comercial (MQILP)
- **yfinance**: API de datos financieros de Yahoo Finance
- **Streamlit**: Framework de aplicaciones web para Python
- **Pandas/NumPy**: Manipulación de datos y cálculos numéricos
- **Matplotlib**: Visualización de datos

---

## 👥 Autores

**Proyecto Cuantitativo - UCEMA**

- Desarrollo de modelos de optimización
- Implementación de sistema de backtesting
- Interfaz web interactiva

---

## 📝 Licencia

Este proyecto es de uso académico. Para uso comercial, contactar a los autores.

---

## 📧 Contacto

Para preguntas, sugerencias o reporte de bugs, abrir un issue en el repositorio de GitHub.
