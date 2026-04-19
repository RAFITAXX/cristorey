# =============================================================================
# data/robot_data.py - Módulo de Datos del Robot
# Tarea 1 - Programación 2 (PUCV) - EIE 434
# =============================================================================

import numpy as np


def cargar_experimentos():
    """
    Retorna un diccionario de diccionarios con los resultados de las
    Tablas 6, 7 y 8 del paper (PPO vs PPO-Mask en distintos ambientes y rutas).
    Llaves internas: politica, ambiente, ruta, ISE, IAE, ITSE, ITAE,
                     tiempo_s, pasos, reward_medio
    """
    experimentos = {
        # ── Tabla 6: Ambiente REAL ────────────────────────────────────────────
        # Ruta Simple
        "exp1": {
            "politica": "PPO",
            "ambiente": "real",
            "ruta": "simple",
            "ISE": 434.99,
            "IAE": 135.93,
            "ITSE": 6932.79,
            "ITAE": 2601.61,
            "tiempo_s": None,
            "pasos": None,
            "reward_medio": None,
        },
        "exp2": {
            "politica": "PPO-Mask",
            "ambiente": "real",
            "ruta": "simple",
            "ISE": 362.85,
            "IAE": 128.92,
            "ITSE": 5869.30,
            "ITAE": 2669.86,
            "tiempo_s": None,
            "pasos": None,
            "reward_medio": None,
        },
        # Ruta Compleja
        "exp3": {
            "politica": "PPO",
            "ambiente": "real",
            "ruta": "compleja",
            "ISE": 576.43,
            "IAE": 178.21,
            "ITSE": 9841.57,
            "ITAE": 3582.44,
            "tiempo_s": None,
            "pasos": None,
            "reward_medio": None,
        },
        "exp4": {
            "politica": "PPO-Mask",
            "ambiente": "real",
            "ruta": "compleja",
            "ISE": 481.20,
            "IAE": 169.03,
            "ITSE": 8203.91,
            "ITAE": 3401.77,
            "tiempo_s": None,
            "pasos": None,
            "reward_medio": None,
        },

        # ── Tabla 7: Ambiente SIMULADO ────────────────────────────────────────
        # Ruta Simple
        "exp5": {
            "politica": "PPO",
            "ambiente": "simulado",
            "ruta": "simple",
            "ISE": 312.74,
            "IAE": 98.36,
            "ITSE": 4821.63,
            "ITAE": 1843.22,
            "tiempo_s": None,
            "pasos": None,
            "reward_medio": None,
        },
        "exp6": {
            "politica": "PPO-Mask",
            "ambiente": "simulado",
            "ruta": "simple",
            "ISE": 260.88,
            "IAE": 93.41,
            "ITSE": 4102.75,
            "ITAE": 1901.54,
            "tiempo_s": None,
            "pasos": None,
            "reward_medio": None,
        },
        # Ruta Compleja
        "exp7": {
            "politica": "PPO",
            "ambiente": "simulado",
            "ruta": "compleja",
            "ISE": 418.52,
            "IAE": 131.74,
            "ITSE": 7023.18,
            "ITAE": 2718.96,
            "tiempo_s": None,
            "pasos": None,
            "reward_medio": None,
        },
        "exp8": {
            "politica": "PPO-Mask",
            "ambiente": "simulado",
            "ruta": "compleja",
            "ISE": 349.61,
            "IAE": 124.88,
            "ITSE": 5912.44,
            "ITAE": 2583.71,
            "tiempo_s": None,
            "pasos": None,
            "reward_medio": None,
        },

        # ── Tabla 8: Comparativa de entrenamiento ────────────────────────────
        # (incluye tiempo_s, pasos y reward_medio; sin índices de error)
        "exp9": {
            "politica": "PPO",
            "ambiente": "entrenamiento",
            "ruta": "simple",
            "ISE": None,
            "IAE": None,
            "ITSE": None,
            "ITAE": None,
            "tiempo_s": 3821.5,
            "pasos": 500000,
            "reward_medio": 142.37,
        },
        "exp10": {
            "politica": "PPO-Mask",
            "ambiente": "entrenamiento",
            "ruta": "simple",
            "ISE": None,
            "IAE": None,
            "ITSE": None,
            "ITAE": None,
            "tiempo_s": 4103.2,
            "pasos": 500000,
            "reward_medio": 178.94,
        },
        "exp11": {
            "politica": "PPO",
            "ambiente": "entrenamiento",
            "ruta": "compleja",
            "ISE": None,
            "IAE": None,
            "ITSE": None,
            "ITAE": None,
            "tiempo_s": 5214.8,
            "pasos": 750000,
            "reward_medio": 118.62,
        },
        "exp12": {
            "politica": "PPO-Mask",
            "ambiente": "entrenamiento",
            "ruta": "compleja",
            "ISE": None,
            "IAE": None,
            "ITSE": None,
            "ITAE": None,
            "tiempo_s": 5587.1,
            "pasos": 750000,
            "reward_medio": 154.83,
        },
    }

    return experimentos


def generar_trayectoria_ideal(waypoints, puntos_por_segmento=100):
    """
    Genera arreglos de puntos intermedios que unen los waypoints dados.

    Parámetros
    ----------
    waypoints : list[list[float]]
        Lista de coordenadas [x, y].
    puntos_por_segmento : int
        Número de puntos interpolados entre cada par de waypoints.

    Retorna
    -------
    np.ndarray, np.ndarray
        Arreglos x_ideal e y_ideal.
    """
    x_ideal = []
    y_ideal = []

    for i in range(len(waypoints) - 1):
        x_start, y_start = waypoints[i]
        x_end,   y_end   = waypoints[i + 1]

        x_seg = np.linspace(x_start, x_end, puntos_por_segmento)
        y_seg = np.linspace(y_start, y_end, puntos_por_segmento)

        x_ideal.extend(x_seg)
        y_ideal.extend(y_seg)

    return np.array(x_ideal), np.array(y_ideal)


def simular_lidar(n_sectores=36, d_min=0.5, d_max=30.0):
    """
    Genera una lectura simulada del sensor RPLIDAR S2.

    Parámetros
    ----------
    n_sectores : int   Número de sectores angulares.
    d_min      : float Distancia mínima del sensor (m).
    d_max      : float Distancia máxima del sensor (m).

    Retorna
    -------
    angulos_deg      : np.ndarray  Ángulos en grados (0 – 360).
    distancias       : np.ndarray  Distancias reales simuladas (m).
    distancias_norm  : np.ndarray  Distancias normalizadas [0, 1].
    """
    # 1. Arreglo de ángulos
    angulos_deg = np.linspace(0, 360, n_sectores, endpoint=False)

    # 2. Distancias aleatorias uniformes
    distancias = np.random.uniform(d_min, d_max, n_sectores)

    # 3. Simulación de obstáculo cercano en sectores [5:9]
    distancias[5:9] = np.random.uniform(0.5, 2.0, 4)

    # 4. Normalización
    distancias_norm = (distancias - d_min) / (d_max - d_min)

    return angulos_deg, distancias, distancias_norm
    return angulos_deg, distancias, distancias_norm 
