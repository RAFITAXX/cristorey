# -----------------------------
# analisis.py
# -----------------------------
import numpy as np


def comparar_rendimiento(datos: list) -> dict:
    # Se analiza el historial de datos de la simulación y calculamos las métricas de rendimiento por cada robot usando NumPy.
    #Parámetro: datos: lista de filas con formato [paso, nombre, x, y, bateria, basura_recolectada]
    #Esto retornara un diccionario con el nombre del robot como su clave y la subdirección con consumo de bataria, basura total y eficiencia

    # 1.Convertimos la lista a un array de NumPy
    matriz = np.array(datos, dtype=object)

    # 2.Se obitienen los nombres únicos de cada robot (columna índice 1)
    nombres_unicos = np.unique(matriz[:, 1])

    # 3. Se crea el diccionario de los resultados
    resultados = {}

    # 4.Procesamos cada robot
    for nombre in nombres_unicos:
        #Máscara booleana para poder filtrar las filas del robot actual
        mascara = matriz[:, 1] == nombre
        filas_robot = matriz[mascara]

        #Se extraen las columnas de batería (índice 4) y basura (índice 5) como valores float
        col_bateria = filas_robot[:, 4].astype(float)
        col_basura = filas_robot[:, 5].astype(float)

        # El consumo de batería será la diferencia entre la batería inicial teórica y el último valor
        consumo_bateria = 100.0 - col_bateria[-1]

        # La Basura total es el último valor acumulado del historial
        basura_total = col_basura[-1]

        # La eficiencia es la basura total dividida por el consumo de batería evitando la división por cero
        if consumo_bateria == 0:
            eficiencia = 0.0
        else:
            eficiencia = basura_total / consumo_bateria

        # almacenamos los resultados del robot
        resultados[nombre] = {
            'consumo_bateria': float(consumo_bateria),
            'basura_total': float(basura_total),
            'eficiencia': float(eficiencia)
        }

    return resultados
