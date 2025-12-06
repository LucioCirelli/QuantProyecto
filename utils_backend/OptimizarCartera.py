import pyomo.environ as pyo
import pandas as pd


def minimizar_riesgo_custom(mu_dict, Sigma_dict, delta_dict, aversion=2, 
                           size_portfolio_max=15, size_portfolio_min=10,
                           peso_maximo=0.15, peso_minimo=0.015):
    """Versión parametrizable de minimizar_riesgo"""
    modelo = pyo.ConcreteModel()

    # --- Sets ---
    activos = list(mu_dict.keys())
    modelo.S = pyo.Set(initialize=activos)  # activos

    # --- Parámetros ---
    # Hiperparametros configurables
    modelo.size_portfolio_max = pyo.Param(initialize=size_portfolio_max)
    modelo.size_portfolio_min = pyo.Param(initialize=size_portfolio_min)
    modelo.peso_maximo = pyo.Param(initialize=peso_maximo)
    modelo.peso_minimo = pyo.Param(initialize=peso_minimo)
    modelo.aversion_riesgo = pyo.Param(initialize=aversion)  # aversión al riesgo

    # Estimados
    modelo.mu = pyo.Param(modelo.S, initialize=mu_dict)
    modelo.delta = pyo.Param(modelo.S, initialize=delta_dict)
    modelo.Sigma = pyo.Param(modelo.S, modelo.S, initialize=Sigma_dict, default=0)

    # --- Variables ---
    modelo.PESO_ACTIVO = pyo.Var(modelo.S, within=pyo.NonNegativeReals, bounds=(0, 1))
    modelo.INCLUIDO = pyo.Var(modelo.S, within=pyo.Binary)

    # --- Función objetivo: minimizar robusta---
    def obj_rule(model):
        # Término lineal (rendimientos esperados - penalización por incertidumbre)
        retorno_esperado = sum((model.mu[i] - model.delta[i]) * model.PESO_ACTIVO[i] for i in model.S)
        # Término cuadrático (riesgo)
        riesgo = sum(model.Sigma[i, j] * model.PESO_ACTIVO[i] * model.PESO_ACTIVO[j] for i in model.S for j in model.S)
        return (retorno_esperado - modelo.aversion_riesgo*riesgo)  # negativamos porque Pyomo minimiza
    modelo.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)


    # --- Restricciones ---
    @modelo.Constraint()
    def suma_pesos(model):
        return sum(model.PESO_ACTIVO[i] for i in model.S) == 1

    # 3. Cantidad de activos seleccionados
    @modelo.Constraint()
    def max_activos(model):
        return sum(model.INCLUIDO[i] for i in model.S) >= model.size_portfolio_min

    @modelo.Constraint()
    def min_activos(model):
        return sum(model.INCLUIDO[i] for i in model.S) <= model.size_portfolio_max

    # 4. Si no está incluido, su peso debe ser 0
    @modelo.Constraint(modelo.S)
    def limite_peso(model, i):
        return model.PESO_ACTIVO[i] <= model.INCLUIDO[i]* model.peso_maximo

    # Peso minimo
    @modelo.Constraint(modelo.S)
    def r_peso_minimo(model, i):
        return model.PESO_ACTIVO[i] >= model.INCLUIDO[i]* model.peso_minimo

    # --- Resolver ---
    try:
        solver = pyo.SolverFactory('gurobi')
        #set gap
        solver.options['MIPGap'] = 0.01
        solver.solve(modelo, tee=False)
        pesos = {i: pyo.value(modelo.PESO_ACTIVO[i]) for i in modelo.S}
        return pesos
    except Exception as e:
        print(f"No se pudo resolver el modelo: {e}")
        print("Se intentará resolver con modelo naive")
        return minimizar_riesgo_naive(mu_dict, Sigma_dict)


def minimizar_riesgo(mu_dict, Sigma_dict, delta_dict, aversion=2):
    """Versión original con parámetros fijos (para compatibilidad)"""
    return minimizar_riesgo_custom(
        mu_dict, Sigma_dict, delta_dict,
        aversion=aversion,
        size_portfolio_max=15,
        size_portfolio_min=10,
        peso_maximo=0.15,
        peso_minimo=0.015
    )


def minimizar_riesgo_naive(mu_dict, Sigma_dict):
    modelo = pyo.ConcreteModel()

    # --- Sets ---
    activos = list(mu_dict.keys())
    modelo.S = pyo.Set(initialize=activos)  # activos

    # --- Parámetros ---
    # Hiperparametros
    modelo.size_portfolio_max = pyo.Param(initialize=15)  # cantidad de activos en la cartera
    modelo.size_portfolio_min = pyo.Param(initialize=10)  # cantidad mínima de activos en la cartera
    modelo.peso_maximo = pyo.Param(initialize=0.15)  # peso máximo por activo
    modelo.peso_minimo = pyo.Param(initialize=0.025)  # peso mínimo por activo
    modelo.aversion_riesgo = pyo.Param(initialize=1)  # aversión al riesgo

    # Estimados
    modelo.mu = pyo.Param(modelo.S, initialize=mu_dict)
    modelo.Sigma = pyo.Param(modelo.S, modelo.S, initialize=Sigma_dict, default=0)

    # --- Variables ---
    modelo.PESO_ACTIVO = pyo.Var(modelo.S, within=pyo.NonNegativeReals, bounds=(0, 1))
    modelo.INCLUIDO = pyo.Var(modelo.S, within=pyo.Binary)

    # --- Función objetivo: minimizar robusta---
    def obj_rule(model):
        # Término lineal (rendimientos esperados - penalización por incertidumbre)
        retorno_esperado = sum((model.mu[i]) * model.PESO_ACTIVO[i] for i in model.S)
        return (retorno_esperado)  # negativamos porque Pyomo minimiza
    modelo.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)


    # --- Restricciones ---
    @modelo.Constraint()
    def suma_pesos(model):
        return sum(model.PESO_ACTIVO[i] for i in model.S) == 1

    # 3. Cantidad de activos seleccionados
    @modelo.Constraint()
    def max_activos(model):
        return sum(model.INCLUIDO[i] for i in model.S) >= model.size_portfolio_min

    @modelo.Constraint()
    def min_activos(model):
        return sum(model.INCLUIDO[i] for i in model.S) <= model.size_portfolio_max

    # 4. Si no está incluido, su peso debe ser 0
    @modelo.Constraint(modelo.S)
    def limite_peso(model, i):
        return model.PESO_ACTIVO[i] <= model.INCLUIDO[i]* model.peso_maximo

    # Peso minimo
    @modelo.Constraint(modelo.S)
    def r_peso_minimo(model, i):
        return model.PESO_ACTIVO[i] >= model.INCLUIDO[i]* model.peso_minimo

    # --- Resolver ---
    try:
        solver = pyo.SolverFactory('gurobi')
        #set gap
        solver.options['MIPGap'] = 0.01
        solver.solve(modelo, tee=True)
        pesos = {i: pyo.value(modelo.PESO_ACTIVO[i]) for i in modelo.S}
        return pesos
    except Exception as e:
        print(f"No se pudo resolver el modelo: {e}")
        return None



if __name__ == "__main__":
    print("Este módulo define la función minimizar_riesgo(mu, Sigma, delta) para optimizar carteras.")