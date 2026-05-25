# -----------------------------
# visualizacion.py
# -----------------------------
import matplotlib.pyplot as plt
import numpy as np


def graficar_recoleccion_vs_bateria(resultados: dict):
    #Generaremos un gráfico de barras agrupadas comparando la basura recolectada con el consumo de batería de cada robot.

    #Los parámetros son:
    #    resultados: diccionario con nombre del robot como clave y métricas como valor.    

    # 1.SE extraen nombres de los robots (categorías del eje X)
    nombres = list(resultados.keys())

    # 2. Extraer listas paralelas de basura total y consumo de batería
    basura_total = [resultados[nombre]['basura_total'] for nombre in nombres]
    consumo_bateria = [resultados[nombre]['consumo_bateria'] for nombre in nombres]

    # 3. Gráfico de barras agrupadas
    x = np.arange(len(nombres))  # Posiciones base en el eje X
    ancho = 0.35                  # Ancho de cada barra

    fig, ax = plt.subplots()

    # Barra de basura desplazada a la izquierda, en verde
    barras_basura = ax.bar(x - ancho / 2, basura_total, ancho,
                           label='Basura Recolectada (kg)', color='green')

    # Barra de batería desplazada a la derecha, en rojo
    barras_bateria = ax.bar(x + ancho / 2, consumo_bateria, ancho,
                            label='Batería Consumida (%)', color='red')

    # 4. Estilo obligatorio del gráfico
    ax.set_title('Rendimiento: Recolección vs Consumo Energético')
    ax.set_ylabel('Cantidad')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.legend()

    # Cuadrícula horizontal para mejor lectura
    ax.grid(axis='y')

    # 5.se muestra el gráfico
    plt.tight_layout()
    plt.show()
