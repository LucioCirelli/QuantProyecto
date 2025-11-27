from cycler import V
import pyomo.environ as pyo
import pickle
import numpy as np
from pyomo.common.timing import report_timing

report_timing()

def modelo_estocastico(nombre_corrida, max_acciones=10, w_minimo=0.05, w_maximo=0.3, rendimiento_minimo=np.log(0.015)):
    """Ejecuta el modelo estocástico con los parámetros dados."""

    # Lectura de datos
    with open(f'Corridas/{nombre_corrida}/inputs_modelo.pkl', 'rb') as f:
        inputs_modelo = pickle.load(f)

    model = pyo.ConcreteModel()

    # Conjuntos
    model.ACCION = pyo.Set(initialize=inputs_modelo['set_acciones'])

    # Parámetros
    model.mu = pyo.Param(model.ACCION, initialize=inputs_modelo['rendimiento_esperado'])
    model.desvio = pyo.Param(model.ACCION, initialize=inputs_modelo['desvio_estandar'])
    model.cov = pyo.Param(model.ACCION, model.ACCION, initialize=inputs_modelo['covarianzas'])
    model.cvar = pyo.Param(model.ACCION,initialize=inputs_modelo['cvar'])
    model.var = pyo.Param(model.ACCION,initialize=inputs_modelo['var'])
    model.probabilidad_perdida = pyo.Param(model.ACCION,initialize=inputs_modelo['probabilidad_perdida'])

    # Escalares
    model.max_acciones = max_acciones
    model.w_minimo = w_minimo
    model.w_maximo = w_maximo
    # model.w_renta_fija = w_renta_fija
    # model.rendimiento_renta_fija = tasa_mensual_renta_fija
    model.rendimiento_minimo_portafolio = rendimiento_minimo
    model.tasa_libre_riesgo = 0.04 / 12

    # Variables
    model.W = pyo.Var(model.ACCION, domain=pyo.NonNegativeReals)
    model.ACTIVAR_ACCION = pyo.Var(model.ACCION, domain=pyo.Binary)
    model.RENDIMIENTO_PORTAFOLIO = pyo.Var(domain=pyo.Reals)
    model.RIESGO_PORTAFOLIO = pyo.Var(domain=pyo.Reals)
    # model.SHARPE = pyo.Var(domain=pyo.Reals)
    model.COSTO_PERDIDA = pyo.Var(domain=pyo.Reals)

    # Restricciones
    @model.Constraint()
    def restriccion_pesos_w(model):
        return sum(model.W[i] for i in model.ACCION) == 1

    @model.Constraint(model.ACCION)
    def restriccion_max_w_accion(model, i):
        return model.W[i] <= model.ACTIVAR_ACCION[i] * model.w_maximo

    @model.Constraint(model.ACCION)
    def restriccion_min_w_accion(model, i):
        return model.W[i] >= model.ACTIVAR_ACCION[i] * model.w_minimo

    @model.Constraint()
    def restriccion_max_acciones(model):
        return sum(model.ACTIVAR_ACCION[i] for i in model.ACCION) <= model.max_acciones

    @model.Constraint()
    def restriccion_rendimiento_portafolio(model):
        return model.RENDIMIENTO_PORTAFOLIO == sum(model.mu[i] * model.W[i] for i in model.ACCION)

    @model.Constraint()
    def restriccion_riesgo_portafolio(model):
        return model.RIESGO_PORTAFOLIO == sum(model.W[i] * model.W[j] * model.cov[i, j] for i in model.ACCION for j in model.ACCION)

    @model.Constraint()
    def restriccion_rendimiento_minimo(model):
        return model.RENDIMIENTO_PORTAFOLIO >= model.rendimiento_minimo_portafolio

    # @model.Constraint()
    # def definicion_sharpe(model):
    #     return model.SHARPE == (model.RENDIMIENTO_PORTAFOLIO - model.tasa_libre_riesgo) / model.RIESGO_PORTAFOLIO

    @model.Constraint()
    def definicion_costo_perdida(model):
        return model.COSTO_PERDIDA == sum(model.probabilidad_perdida[i] * model.cvar[i] * model.W[i] for i in model.ACCION)

    @model.Objective(sense=pyo.minimize)
    def minimizar_riesgo(model):
        return model.RIESGO_PORTAFOLIO + 0.5 * model.COSTO_PERDIDA + 0.01 * sum((1- model.ACTIVAR_ACCION[i]) * model.mu[i] for i in model.ACCION)

    opt = pyo.SolverFactory('gurobi')
    opt.options['TimeLimit'] = 1000
    opt.options['MIPGap'] = 0.05
    results = opt.solve(model, tee=True)

    # ============================================================================
    # POSTPROCESAMIENTO: Generar reporte Excel
    # ============================================================================

    from utils.Postprocesamiento import generar_reporte_excel
    import os

    # Generar reporte en la carpeta de la corrida
    ruta_reporte = os.path.join(f'Corridas/{nombre_corrida}', 'resultados_estocastico.xlsx')
    generar_reporte_excel(model, inputs_modelo, ruta_reporte)

    print(f"\n✅ Modelo resuelto y reporte generado en: Corridas/{nombre_corrida}/")


if __name__ == "__main__":
    nombre_corrida = "20251025_150353_analisis_sp500"  # Cambiar según la corrida deseada
    
    max_acciones = 10
    w_minimo = 0.05
    w_maximo = 0.3
    # w_renta_fija = 0.2
    # tasa_mensual_renta_fija = 0.08/12
    rendimiento_minimo = np.log(0.015)
    
    modelo_estocastico(nombre_corrida, max_acciones, w_minimo, w_maximo, rendimiento_minimo)
