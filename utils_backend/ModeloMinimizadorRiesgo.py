from cycler import V
import pyomo.environ as pyo
import pickle
import numpy as np
from pyomo.common.timing import report_timing

report_timing()

def modelo_minimizador_riesgo(mu_dict, Sigma_dict, delta_dict, cvar_dict=None, var_dict=None, prob_perdida_dict=None, 
                              max_acciones=10, w_minimo=0.05, w_maximo=0.3, rendimiento_minimo=np.log(0.015)):
    """Ejecuta el modelo minimizador de riesgo con los parámetros dados.
    
    Args:
        mu_dict: Dict de rendimientos esperados {ticker: valor}
        Sigma_dict: Dict de matriz de covarianza {(ticker1, ticker2): valor}
        delta_dict: Dict de desviaciones estándar {ticker: valor}
        cvar_dict: Dict de CVaR {ticker: valor} (opcional, se calcula si no se provee)
        var_dict: Dict de VaR {ticker: valor} (opcional, se calcula si no se provee)
        prob_perdida_dict: Dict de probabilidad de pérdida {ticker: valor} (opcional, default 5%)
        max_acciones: Máximo número de acciones
        w_minimo: Peso mínimo por acción
        w_maximo: Peso máximo por acción
        rendimiento_minimo: Rendimiento mínimo del portfolio
    """

    model = pyo.ConcreteModel()

    # Conjuntos
    model.ACCION = pyo.Set(initialize=list(mu_dict.keys()))

    # Parámetros
    model.mu = pyo.Param(model.ACCION, initialize=mu_dict)
    model.desvio = pyo.Param(model.ACCION, initialize=delta_dict)
    model.cov = pyo.Param(model.ACCION, model.ACCION, initialize=Sigma_dict)
    
    # CVaR, VaR y probabilidad_perdida - usar proporcionados o calcular por defecto
    if cvar_dict is None:
        cvar_dict = {ticker: 1.96 * delta_dict[ticker] for ticker in mu_dict.keys()}
    if var_dict is None:
        var_dict = {ticker: 1.65 * delta_dict[ticker] for ticker in mu_dict.keys()}
    if prob_perdida_dict is None:
        prob_perdida_dict = {ticker: 0.05 for ticker in mu_dict.keys()}
    
    model.cvar = pyo.Param(model.ACCION, initialize=cvar_dict)
    model.var = pyo.Param(model.ACCION, initialize=var_dict)
    model.probabilidad_perdida = pyo.Param(model.ACCION, initialize=prob_perdida_dict)

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
        # return model.RIESGO_PORTAFOLIO + 0.5 * model.COSTO_PERDIDA + 0.01 * sum((1- model.ACTIVAR_ACCION[i]) * model.mu[i] for i in model.ACCION)
        return model.RIESGO_PORTAFOLIO + model.COSTO_PERDIDA + 0.1 * sum((1- model.ACTIVAR_ACCION[i]) * model.mu[i] for i in model.ACCION)

    opt = pyo.SolverFactory('gurobi')
    opt.options['TimeLimit'] = 1000
    opt.options['MIPGap'] = 0.02
    results = opt.solve(model, tee=True)
    
    pesos = {i: pyo.value(model.W[i]) for i in model.ACCION}
    return pesos


if __name__ == "__main__":
    nombre_corrida = "20251025_150353_analisis_sp500"  # Cambiar según la corrida deseada
    
    max_acciones = 10
    w_minimo = 0.05
    w_maximo = 0.3
    # w_renta_fija = 0.2
    # tasa_mensual_renta_fija = 0.08/12
    rendimiento_minimo = np.log(0.015)
    
    modelo_minimizador_riesgo(nombre_corrida, max_acciones, w_minimo, w_maximo, rendimiento_minimo)
