# =============================================================================
# processing/cinematicas.py - Módulo de Cinemática del Robot Diferencial
# Tarea 1 - Programación 2 (PUCV) - EIE 434
# =============================================================================
# Implementa el modelo cinemático (Ecs. 1–6 del paper) y funciones de error.
# =============================================================================

import numpy as np

# ── Restricciones físicas del robot (Tabla 1 del paper) ──────────────────────
V_MAX     = 0.8   # m/s   velocidad lineal máxima
OMEGA_MAX = 0.6   # rad/s velocidad angular máxima


def calcular_movimiento(x, y, theta, v, omega, dt=0.1):
    """
    Calcula la nueva pose del robot diferencial tras un paso de tiempo.

    Ecuaciones cinemáticas:
        x_new     = x + v · cos(θ) · dt
        y_new     = y + v · sin(θ) · dt
        theta_new = θ + ω · dt

    Parámetros
    ----------
    x, y   : float  Posición actual (m).
    theta  : float  Orientación actual (rad).
    v      : float  Velocidad lineal (m/s).
    omega  : float  Velocidad angular (rad/s).
    dt     : float  Paso de tiempo (s). Por defecto 0.1 s.

    Retorna
    -------
    tuple (x_nuevo, y_nuevo, theta_nuevo)
    """
    # Saturación de velocidades según Tabla 1
    v     = float(np.clip(v,     -V_MAX,     V_MAX))
    omega = float(np.clip(omega, -OMEGA_MAX, OMEGA_MAX))

    x_nuevo     = x     + v * np.cos(theta) * dt
    y_nuevo     = y     + v * np.sin(theta) * dt
    theta_nuevo = theta + omega * dt

    return x_nuevo, y_nuevo, theta_nuevo


def distancia_al_objetivo(x, y, x_meta, y_meta):
    """
    Calcula la distancia euclidiana entre la posición actual y la meta.

    Parámetros
    ----------
    x, y         : float  Posición actual del robot (m).
    x_meta, y_meta : float  Coordenadas del objetivo (m).

    Retorna
    -------
    float  Distancia en metros.
    """
    return float(np.sqrt((x_meta - x) ** 2 + (y_meta - y) ** 2))


def calcular_error_seguimiento(x_real, y_real, x_ideal, y_ideal):
    """
    Calcula el error de posición punto a punto entre la trayectoria real
    y la trayectoria ideal.

    Si los arreglos tienen distinto tamaño, se usa el tamaño del más corto
    para evitar errores de índice.

    Parámetros
    ----------
    x_real,  y_real  : np.ndarray  Trayectoria ejecutada por el robot.
    x_ideal, y_ideal : np.ndarray  Trayectoria de referencia ideal.

    Retorna
    -------
    np.ndarray  Distancia de error en cada punto (m).
    """
    n = min(len(x_real), len(x_ideal))
    dx = x_real[:n] - x_ideal[:n]
    dy = y_real[:n] - y_ideal[:n]
    return np.sqrt(dx ** 2 + dy ** 2)