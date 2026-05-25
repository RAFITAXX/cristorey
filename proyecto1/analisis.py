# ==========================================
# analisis.py
# ==========================================
import numpy as np


def comparar_rendimiento(datos: list) -> dict:
    """
    Analiza el historial de datos de la simulación y calcula
    métricas de rendimiento por robot usando NumPy.

    Parámetro:
        datos: lista de filas con formato [paso, nombre, x, y, bateria, basura_recolectada]

    Retorna:
        Diccionario con nombre del robot como clave y sub-diccionario con:
        consumo_bateria, basura_total y eficiencia.
    """

    # 1. Convertir la lista a un array de NumPy
    matriz = np.array(datos, dtype=object)

    # 2. Obtener los nombres únicos de los robots (columna índice 1)
    nombres_unicos = np.unique(matriz[:, 1])

    # 3. Diccionario de resultados
    resultados = {}

    # 4. Procesar cada robot
    for nombre in nombres_unicos:
        # Máscara booleana para filtrar filas del robot actual
        mascara = matriz[:, 1] == nombre
        filas_robot = matriz[mascara]

        # Extraer columnas de batería (índice 4) y basura (índice 5) como float
        col_bateria = filas_robot[:, 4].astype(float)
        col_basura = filas_robot[:, 5].astype(float)

        # Consumo de batería: diferencia entre batería inicial teórica y último valor
        consumo_bateria = 100.0 - col_bateria[-1]

        # Basura total: último valor acumulado del historial
        basura_total = col_basura[-1]

        # Eficiencia: basura total / consumo de batería (evitar división por cero)
        if consumo_bateria == 0:
            eficiencia = 0.0
        else:
            eficiencia = basura_total / consumo_bateria

        # Almacenar resultados del robot
        resultados[nombre] = {
            'consumo_bateria': float(consumo_bateria),
            'basura_total': float(basura_total),
            'eficiencia': float(eficiencia)
        }

    return resultados
