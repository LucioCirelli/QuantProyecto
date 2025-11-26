# %%
"""
PIPELINE_COMPLETO.PY - Pipeline integrado con dos modelos y backtesting

Ejecuta automáticamente:
1. Preprocesamiento de datos (opcional)
2. Modelo Estocástico
3. Modelo Robust Optimization
4. Backtesting comparativo

Uso:
    # Corrida completa (con preprocesamiento)
    CONFIG['ejecutar_preprocesamiento'] = True
    python Pipeline_Completo.py
    
    # Usar corrida existente (sin preprocesamiento)
    CONFIG['ejecutar_preprocesamiento'] = False
    python Pipeline_Completo.py
"""

import os
import numpy as np

from OrquestaPreprocesamiento import ejecutar_preprocesamiento

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================


def ejecutar_modelo_estocastico(nombre_corrida, config):
    """
    Ejecuta ModeloEstocastico.py y genera resultados_estocastico.xlsx
    """
    
    params = config['estocastico']
    
    from ModeloEstocastico import modelo_estocastico
    
    modelo_estocastico(
        nombre_corrida=nombre_corrida,
        max_acciones=params['max_acciones'],
        w_minimo=params['w_minimo'],
        w_maximo=params['w_maximo'],
        rendimiento_minimo=params['rendimiento_minimo']
    )
        
    return True

def ejecutar_modelo_robust(nombre_corrida, config):
    """
    Ejecuta ModeloRobustOptimization.py y genera resultados_robust.xlsx
    """

    params = config['robust']
    
    from ModeloRobustOptimization import modelo_robust_optimization
        
    modelo_robust_optimization(
        nombre_corrida=nombre_corrida,
        max_acciones=params['max_acciones'],
        w_minimo=params['w_minimo'],
        w_maximo=params['w_maximo'],
        rendimiento_minimo=params['rendimiento_minimo']
    )
    
    return True


def ejecutar_backtesting(dir_corrida, config):
    """
    Ejecuta backtesting comparando ambos modelos vs S&P 500
    """
    
    from Backtesting import ejecutar_backtesting as run_backtesting
        
    run_backtesting(
        corrida_1=dir_corrida,
        corrida_2=dir_corrida,
        año_inicio=config['año_backtesting'],
        guardar_en=os.path.join(dir_corrida, 'Backtesting'),
        nombre_archivo_1='resultados_estocastico.xlsx',
        nombre_archivo_2='resultados_robust.xlsx'
    )
        
    return True


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def ejecutar_pipeline_completo(config, correr_preprocess = True):
    """
    Ejecuta pipeline completo: preprocesamiento + 2 modelos + backtesting
    
    Parámetros:
        config : dict, configuración del pipeline
    """

    # Crear carpeta
    dir_corrida = os.path.join('Corridas', config['nombre_corrida'])
    os.makedirs(dir_corrida, exist_ok=True)
    
    print(f"\n📁 Corrida: {dir_corrida}")
    
    # Preprocesamiento opcional
    if correr_preprocess:
        ejecutar_preprocesamiento(dir_corrida, config)
    
    # Modelos
    ejecutar_modelo_estocastico(config['nombre_corrida'], config)
    ejecutar_modelo_robust(config['nombre_corrida'], config)
    
    ejecutar_backtesting(dir_corrida, config)
    
    
    print('PROCESO FINALIZADO')


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    nombre_corrida = 'Corrida'
    start_year = 2019
    end_year = 2024
    alpha_ewma = 0.94
    ejecutar_preprocesamiento(nombre_corrida, start_year, end_year, alpha_ewma)
    
    CONFIG = {
        'nombre_corrida': 'Corrida',
        'start_year': 2019,
        'end_year': 2024,
        'alpha_ewma': 0.94,
        'max_activos': None,  # None = todos
        'año_backtesting': 2025,  # Año desde el cual hacer backtesting
        'ejecutar_preprocesamiento': False,  # False = usar inputs_modelo.pkl existente
        
        # Parámetros Modelo Estocástico
        'estocastico': {
            'max_acciones': 10,
            'w_minimo': 0.05,
            'w_maximo': 0.2,
            'rendimiento_minimo': np.log(1.015)  # 1.5% mensual en log
        },
        
        # Parámetros Modelo Robust Optimization
        'robust': {
            'max_acciones': 30,
            'w_minimo': 0.05,
            'w_maximo': 0.2,
            'rendimiento_minimo': -3  # Permite más flexibilidad
        }
    }

    ejecutar_pipeline_completo(CONFIG, correr_preprocess=CONFIG['ejecutar_preprocesamiento'])
