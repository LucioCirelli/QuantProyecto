# 📊 Portfolio Optimizer - Versión Simplificada

## 🎯 Aplicación de 2 Tabs

Aplicación Streamlit **ultra-simplificada** para backtesting de portfolios con rebalanceo mensual.

Implementa **EXACTAMENTE** la metodología de Franco (DinamicMVO.ipynb).

---

## 🚀 Cómo Usar

### 1. Ejecutar la aplicación

```powershell
cd Pipeline_Modelos_Propios
streamlit run app_simple.py
```

### 2. Tab 1: Preprocesamiento

- **Cargar datos**: Selecciona rango de años (ej: 2010-2025)
- **Configurar parámetros**: Ajusta parámetros de cada modelo
  - Estocástico: max_acciones, w_minimo, w_maximo
  - Robust: max_acciones, w_minimo, w_maximo
  - Franco: aversion_riesgo

### 3. Tab 2: Backtesting & Resultados

- **Configurar backtesting**:
  - Fecha inicio (ej: 2024-01-01)
  - Ventana histórica (ej: 100 meses)
  - Frecuencia rebalanceo (ej: 1 mes)

- **Seleccionar modelos**: Marca 1, 2 o 3 modelos a comparar

- **Ejecutar**: Click en "EJECUTAR BACKTESTING" → Caja negra

- **Resultados**: Gráficos IDÉNTICOS a Franco para cada modelo

---

## 📁 Estructura de Archivos (LIMPIA)

```
Pipeline_Modelos_Propios/
│
├── app_simple.py                  ← APLICACIÓN PRINCIPAL (2 tabs)
├── funciones_backtesting.py       ← TODA la lógica de Franco (backtesting + gráficos)
│
├── ModeloEstocastico.py           ← Modelo Estocástico (CVaR)
├── ModeloRobustOptimization.py    ← Modelo Robust Optimization
│
├── utils/
│   ├── CargarDatos.py             ← Descarga S&P 500 + SPY
│   └── __init__.py
│
├── Corridas/                      ← Outputs temporales de modelos
├── README_APP_SIMPLE.md           ← Esta guía
└── requirements.txt               ← Dependencias
```

**Total: Solo 5 archivos Python principales** (ultra-simplificado)

---

## 🔧 Funcionalidades

### Tab 1: Preprocesamiento
- ✅ Descarga automática de S&P 500 desde Wikipedia
- ✅ Descarga de SPY (benchmark) desde yfinance
- ✅ Cálculo de retornos logarítmicos mensuales
- ✅ Configuración de parámetros por modelo

### Tab 2: Backtesting & Resultados
- ✅ Rebalanceo mensual con ventana móvil
- ✅ Cálculo de inputs método Franco: `μ = 0.5*media + 0.5*momentum`
- ✅ Ejecución de 1, 2 o 3 modelos simultáneos
- ✅ Comparación automática vs SPY
- ✅ Gráficos de 6 paneles (IDÉNTICOS a DinamicMVO.ipynb):
  1. Retornos acumulados con áreas de outperformance
  2. Drawdown chart
  3. Retornos por período (barras)
  4. Distribución de retornos
  5. Tabla de métricas completa
- ✅ Métricas calculadas:
  - Retorno total y anualizado
  - Volatilidad anualizada
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Information Ratio
  - Win Rate
  - Maximum Drawdown

---

## 🎨 Modelos Disponibles

### 1. Estocástico (CVaR)
- Minimiza riesgo + CVaR + costo de pérdida
- Usa programación estocástica
- Restricciones: max_acciones, w_minimo, w_maximo

### 2. Robust Optimization
- Optimización robusta con intervalos de incertidumbre
- Considera worst-case scenarios
- Restricciones: max_acciones, w_minimo, w_maximo

### 3. Franco (MVO Dinámico)
- Mean-Variance Optimization con penalización robusta
- **Modo dinámico**: Minimiza turnover con pesos anteriores
- Parámetros fijos en el modelo:
  - Max activos: 30
  - Min activos: 20
  - Peso máximo: 10%
  - Peso mínimo: 1.5%
  - Turnover limit: 75%

---

## 📊 Metodología (Franco)

### Cálculo de Inputs (Cada Rebalanceo)

1. **Ventana móvil**: Últimos N meses (ej: 100)

2. **Rendimiento esperado**:
   ```python
   μ = 0.5 * media_historica + 0.5 * momentum_3_meses
   ```

3. **Matriz de covarianzas**:
   ```python
   Σ = cov(retornos_históricos)
   ```

4. **Delta (incertidumbre)**:
   ```python
   δ = 1.96 * σ / √T
   ```

### Rebalanceo Mensual

```
Para cada mes t:
  1. Calcular inputs con ventana [t-window, t]
  2. Ejecutar modelo con pesos anteriores (si existen)
  3. Obtener nuevos pesos óptimos
  4. Calcular retorno del mes siguiente
  5. Comparar vs SPY
```

### Acumulación de Retornos

```python
retorno_acumulado = exp(Σ retornos_log) - 1
```

---

## 🧪 Ejemplo de Uso

```python
# Tab 1: Cargar datos 2010-2025
# Tab 1: Configurar aversion_riesgo = 2.0 para Franco

# Tab 2: Configurar backtesting
start_date = "2024-01-01"
window = 100 meses
rebalance = 1 mes

# Tab 2: Seleccionar modelos
✓ Franco (MVO Dinámico)

# Tab 2: Ejecutar
→ Click "EJECUTAR BACKTESTING"

# Resultado: Gráficos + Métricas
Retorno Total: +15.23%
Sharpe Ratio: 1.85
Max Drawdown: -8.45%
Win Rate: 58.3%
```

---

## 🔍 Diferencias con Versión Anterior

### ANTES (Complejo):
- ❌ 4 tabs separados
- ❌ Múltiples archivos utils_front/
- ❌ Imports anidados complejos
- ❌ Backtesting separado de resultados
- ❌ Gráficos plotly diferentes a Franco

### AHORA (Simple):
- ✅ 2 tabs únicos
- ✅ 2 archivos principales (app + funciones)
- ✅ Imports directos simples
- ✅ Backtesting + resultados juntos
- ✅ Gráficos matplotlib IDÉNTICOS a Franco

---

## 🎯 Principios de Diseño

1. **Simplicidad**: Solo lo esencial
2. **Caja negra**: Usuario clickea y ve resultados
3. **Fidelidad**: Metodología 100% de Franco
4. **Mantenibilidad**: Código legible y compacto

---

## 📝 Notas Técnicas

### Modelo Dinámico (Franco)

Solo el modelo de Franco usa modo dinámico con restricción de turnover:

```python
if pesos_anteriores is not None:
    # Usa minimizar_riesgo_dinamico() 
    # con restricción: w_nuevo >= w_anterior * (1 - turnover_limit)
else:
    # Primera iteración: minimizar_riesgo() clásico
```

Los otros modelos (Estocástico, Robust) no tienen modo dinámico implementado.

### Gestión de NaNs

Todos los inputs se limpian automáticamente:
- NaNs en μ → mediana
- Inf en σ → media
- NaNs en Σ → 0 (off-diagonal) o σ² (diagonal)

---

## 🐛 Solución de Problemas

### Error: "No se pudo cargar datos"
- Verificar conexión a internet
- Wikipedia puede estar bloqueada → usar VPN

### Error: "Gurobi no encontrado"
- Instalar Gurobi: `pip install gurobipy`
- Obtener licencia académica gratuita

### Error: "Import error Pipeline_Franco"
- Verificar que existe `../Pipeline_Franco/`
- Verificar archivos: `OptimizarCartera.py`, `OptimizarCarteraDinamico.py`

---

## 📚 Referencias

- **Franco DinamicMVO.ipynb**: Notebook original con metodología completa
- **Franco OptimizarCartera.py**: Implementación del modelo MVO
- **Franco OptimizarCarteraDinamico.py**: Implementación con turnover

---

## ✅ Checklist de Testing

- [ ] Cargar datos 2010-2025
- [ ] Configurar parámetros de Franco
- [ ] Ejecutar backtesting desde 2024-01-01
- [ ] Verificar gráficos (6 paneles)
- [ ] Verificar métricas (11 métricas)
- [ ] Comparar vs SPY
- [ ] Probar con 2 modelos simultáneos
- [ ] Probar con 3 modelos simultáneos

---

**Autor**: Adaptado de Franco DinamicMVO  
**Fecha**: Noviembre 2025  
**Versión**: 2.0 Simplificada
