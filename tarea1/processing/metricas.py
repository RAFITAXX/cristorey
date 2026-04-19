# =============================================================================
# processing/metricas.py - Módulo de Métricas de Control
# Tarea 1 - Programación 2 (PUCV) - EIE 434
# =============================================================================
# Implementa los índices de error de control discretos (Ecs. 11–14 del paper).
# Restricción: NO se usan ciclos for para las sumatorias; se usa np.sum.
# =============================================================================

import numpy as np


def calcular_IAE(errores, dt):
    """
    Integral Absolute Error (IAE).
    IAE = Σ |e(i)| · dt

    Parámetros
    ----------
    errores : np.ndarray  Arreglo de errores de posición.
    dt      : float       Paso de tiempo (s).

    Retorna
    -------
    float
    """
    return float(np.sum(np.abs(errores)) * dt)


def calcular_ISE(errores, dt):
    """
    Integral Square Error (ISE).
    ISE = Σ e(i)² · dt

    Parámetros
    ----------
    errores : np.ndarray  Arreglo de errores de posición.
    dt      : float       Paso de tiempo (s).

    Retorna
    -------
    float
    """
    return float(np.sum(errores ** 2) * dt)


def calcular_ITAE(errores, dt):
    """
    Integral Time-weighted Absolute Error (ITAE).
    ITAE = Σ t(i) · |e(i)| · dt

    Parámetros
    ----------
    errores : np.ndarray  Arreglo de errores de posición.
    dt      : float       Paso de tiempo (s).

    Retorna
    -------
    float
    """
    t = np.arange(len(errores)) * dt
    return float(np.sum(t * np.abs(errores)) * dt)


def calcular_ITSE(errores, dt):
    """
    Integral Time-weighted Square Error (ITSE).
    ITSE = Σ t(i) · e(i)² · dt

    Parámetros
    ----------
    errores : np.ndarray  Arreglo de errores de posición.
    dt      : float       Paso de tiempo (s).

    Retorna
    -------
    float
    """
    t = np.arange(len(errores)) * dt
    return float(np.sum(t * errores ** 2) * dt)


def calcular_todas_las_metricas(errores, dt):
    """
    Calcula las cuatro métricas de control y las retorna en un diccionario.

    Parámetros
    ----------
    errores : np.ndarray  Arreglo de errores de posición.
    dt      : float       Paso de tiempo (s).

    Retorna
    -------
    dict con llaves "ISE", "IAE", "ITSE", "ITAE" (valores redondeados a 2 dec.)
    """
    return {
        "ISE":  round(calcular_ISE(errores, dt),  2),
        "IAE":  round(calcular_IAE(errores, dt),  2),
        "ITSE": round(calcular_ITSE(errores, dt), 2),
        "ITAE": round(calcular_ITAE(errores, dt), 2),
    }


def calcular_mejora(valor_ppo, valor_mask):
    """
    Calcula la reducción porcentual del error al pasar de PPO a PPO-Mask.
    mejora = (V_base - V_nuevo) / V_base × 100

    Parámetros
    ----------
    valor_ppo  : float  Métrica del controlador base (PPO).
    valor_mask : float  Métrica del controlador mejorado (PPO-Mask).

    Retorna
    -------
    float  Porcentaje de mejora (positivo → PPO-Mask es mejor).
    """
    if valor_ppo == 0:
        return 0.0
    return round((valor_ppo - valor_mask) / valor_ppo * 100, 2)