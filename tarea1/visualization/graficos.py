
import os
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt


CARPETA_RESULTADOS = "resultados_graficos"


def _asegurar_carpeta(carpeta=CARPETA_RESULTADOS):
    
    os.makedirs(carpeta, exist_ok=True)



#(a) plot_metricas(diccionario_experimentos, ambiente, ruta)
def plot_metricas(diccionario_experimentos, ambiente, ruta):
   
    _asegurar_carpeta()


    datos_ppo  = None
    datos_mask = None

    for _, datos in diccionario_experimentos.items():
        if datos["ambiente"] == ambiente and datos["ruta"] == ruta:
            if datos["politica"] == "PPO":
                datos_ppo = datos
            elif datos["politica"] == "PPO-Mask":
                datos_mask = datos

    if datos_ppo is None or datos_mask is None:
        print(f"[AVISO] No se encontraron datos para ambiente='{ambiente}', ruta='{ruta}'")
        return

    metricas   = ["ISE", "IAE", "ITSE", "ITAE"]
    val_ppo    = [datos_ppo[m]  for m in metricas]
    val_mask   = [datos_mask[m] for m in metricas]

    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f"Índices de error — {ambiente} | {ruta} (Tabla 6)", fontsize=13)

    colores = ["#4C72B0", "#C44E52"]   # azul PPO, rojo PPO-Mask

    for ax, metrica, v_ppo, v_mask in zip(axes, metricas, val_ppo, val_mask):
        bars = ax.bar(["PPO", "PPO-Mask"], [v_ppo, v_mask], color=colores, width=0.5)
        ax.set_title(metrica, fontsize=12, fontweight="bold")
        ax.set_ylabel("Valor del Índice")
        ax.set_ylim(0, max(v_ppo, v_mask) * 1.2)
        # Etiquetas sobre las barras
        for bar, val in zip(bars, [v_ppo, v_mask]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(v_ppo, v_mask) * 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()

    nombre_archivo = os.path.join(
        CARPETA_RESULTADOS,
        f"metricas_{ambiente}_{ruta}.png",
    )
    plt.savefig(nombre_archivo, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" > Gráfico guardado: {nombre_archivo}")

#(b) plot_lidar(angulos, distancias, distancias_norm)
def plot_lidar(angulos, distancias, distancias_norm):
   
    _asegurar_carpeta()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    
    colores_scatter = [
        "red" if d < 2.0 else "green" for d in distancias
    ]
    ax1.scatter(angulos, distancias, c=colores_scatter, s=60, zorder=3)
    ax1.set_title("¿A qué distancia están los objetos?\n(Eje X: Ángulo | Eje Y: Metros)")
    ax1.set_xlabel("Ángulo de giro (0-360°)")
    ax1.set_ylabel("Distancia detectada (m)")
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, None)
    ax1.grid(True, alpha=0.3)

    
    colores_line = [
        "red" if d < (2.0 - 0.5) / (30.0 - 0.5) else "#CC0000"
        for d in distancias_norm
    ]
    ax2.plot(range(len(distancias_norm)), distancias_norm,
             color="red", linewidth=1.5, marker="o", markersize=3)
    ax2.set_title("Datos Normalizados\n(Lo que procesa la IA)")
    ax2.set_xlabel("Sectores del sensor")
    ax2.set_ylabel("Valor (0.0 a 1.0)")
    ax2.set_xlim(0, len(distancias_norm))
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    nombre_archivo = os.path.join(CARPETA_RESULTADOS, "mapa_lidar.png")
    plt.savefig(nombre_archivo, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" > Gráfico guardado: {nombre_archivo}")

#(c) plot_trayectorias(x_ppo, y_ppo, x_mask, y_mask, waypoints, nombre)
def plot_trayectorias(x_ppo, y_ppo, x_mask, y_mask, waypoints, nombre):
    
    _asegurar_carpeta()

    fig, ax = plt.subplots(figsize=(7, 7))

    # Trayectorias
    ax.plot(x_ppo,  y_ppo,  color="#4C72B0", linewidth=1.0,
            alpha=0.8, label="Trayectoria PPO")
    ax.plot(x_mask, y_mask, color="#C44E52", linewidth=1.0,
            linestyle="--", alpha=0.8, label="Trayectoria PPO-Mask")

    # Waypoints como cuadrados negros
    wx = [wp[0] for wp in waypoints]
    wy = [wp[1] for wp in waypoints]
    ax.scatter(wx, wy, marker="s", color="black", s=80, zorder=5,
               label="Waypoints (Metas)")

    ax.set_title(f"Comparación de Navegación: Ruta {nombre.capitalize()}")
    ax.set_xlabel("Posición X (metros)")
    ax.set_ylabel("Posición Y (metros)")
    ax.legend(loc="upper right", fontsize=9)
    ax.axis("equal")   # Restricción geométrica obligatoria
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    nombre_archivo = os.path.join(
        CARPETA_RESULTADOS,
        f"trayectorias_{nombre}.png",
    )
    plt.savefig(nombre_archivo, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" > Gráfico guardado: {nombre_archivo}")
