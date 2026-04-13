import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metricas(diccionario_experimentos, ambiente, ruta):
    """
    Genera una figura con 4 subplots lineales comparando PPO vs PPO-Mask[cite: 157, 158].
    """
    # 1. Filtrado de datos según ambiente y ruta [cite: 160]
    politicas = ["PPO", "PPO-Mask"]
    metricas = ["ISE", "IAE", "ITSE", "ITAE"]
    
    # Preparamos los datos para las barras
    valores_grafico = {m: [] for m in metricas}
    
    for pol in politicas:
        # Buscamos el experimento que coincida con la política, ambiente y ruta
        encontrado = False
        for exp in diccionario_experimentos.values():
            if exp["politica"] == pol and exp["ambiente"] == ambiente and exp["ruta"] == ruta:
                for m in metricas:
                    valores_grafico[m].append(exp[m])
                encontrado = True
                break
        if not encontrado: # Si no hay datos, ponemos 0 para no romper el gráfico
            for m in metricas:
                valores_grafico[m].append(0)

    # 2. Creación de la figura (1 fila, 4 columnas) [cite: 158]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    colores = ['#3498db', '#e74c3c'] # Azul para PPO, Rojo para PPO-Mask
    
    for i, m in enumerate(metricas):
        axs[i].bar(politicas, valores_grafico[m], color=colores)
        axs[i].set_title(m)
        axs[i].set_ylabel("Valor del índice")
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle(f"Indices de error --- {ambiente} | {ruta}")
    plt.tight_layout()

    # 3. Automatización de guardado [cite: 162]
    folder = "resultados_graficos"
    os.makedirs(folder, exist_ok=True) # Crea la carpeta si no existe 
    
    path = os.path.join(folder, f"metricas_{ambiente}_{ruta}.png")
    plt.savefig(path, dpi=300) # Guarda con alta calidad [cite: 155]
    plt.close()

def plot_lidar(angulos, distancias, distancias_norm):
    """
    Visualiza la percepción del robot (Real vs IA)[cite: 167, 169].
    """
    plt.figure(figsize=(12, 5))

    # Subplot 1: Visualización "Humana" (Real) [cite: 170, 171]
    plt.subplot(1, 2, 1)
    plt.scatter(angulos, distancias, c=distancias, cmap='winter')
    plt.title("¿A qué distancia están los objetos?\n(Eje X: Ángulo | Eje Y: Metros)")
    plt.xlabel("Ángulo de giro (0-360°)")
    plt.ylabel("Distancia detectada (m)")

    # Subplot 2: Visualización "IA" (Normalizada) [cite: 172]
    plt.subplot(1, 2, 2)
    plt.plot(range(len(distancias_norm)), distancias_norm, 'r.-')
    plt.title("Datos Normalizados\n(Lo que procesa la IA)")
    plt.xlabel("Sectores del sensor")
    plt.ylabel("Valor (0.0 a 1.0)")

    # Guardado automático
    folder = "resultados_graficos"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, "mapa_lidar.png"), dpi=300)
    plt.close()

def plot_trayectorias(x_ppo, y_ppo, x_mask, y_mask, waypoints, nombre):
    """
    Genera el mapa de navegación comparativo[cite: 173].
    """
    plt.figure(figsize=(8, 8))
    
    # Trayectorias [cite: 174]
    plt.plot(x_ppo, y_ppo, label="Trayectoria PPO", alpha=0.6)
    plt.plot(x_mask, y_mask, label="Trayectoria PPO-Mask", linestyle='--', alpha=0.8)
    
    # Waypoints como cuadrados negros [cite: 174]
    wp = np.array(waypoints)
    plt.scatter(wp[:, 0], wp[:, 1], color='black', marker='s', label="Waypoints (Metas)")
    
    # Restricción Geométrica OBLIGATORIA 
    plt.axis('equal') 
    
    plt.title(f"Comparación de Navegación: {nombre}")
    plt.xlabel("Posición X (metros)")
    plt.ylabel("Posición Y (metros)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Guardado automático
    folder = "resultados_graficos"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"trayectoria_{nombre}.png"), dpi=300)
    plt.close()
