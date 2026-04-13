import numpy as np

def cargar_experimentos():
    """Carga un diccionario de diccionarios con los datos de los experimentos del paper.
    """
    experimentos = {

 # Tabla 6: Experimento simple
 " exp1 ": {" politica ": " PPO", " ambiente ": " real ", " ruta ": " simple ", " ISE ": 434.99 , " IAE":
135.93 , " ITSE ": 6932.79 , " ITAE ": 2601.61 , " tiempo_s ": None , " pasos ": None , " reward_medio ": None } ,
 " exp2 ": {" politica ": "PPO - Mask ", " ambiente ": " real ", " ruta ": " simple ", " ISE ": 362.85 , " IAE ":
128.92 , " ITSE ": 5869.30 , " ITAE ": 2669.86 , " tiempo_s ": None , " pasos ": None , " reward_medio ": None },
.... }
    
    return experimentos

def generar_trayectoria_ideal(waypoints, puntos_por_segmento=100):
    """
    Genera puntos intermedios entre waypoints usando interpolación lineal.
    """
    x_ideal = []
    y_ideal = []
    
    # Iteramos sobre pares de puntos consecutivos [cite: 132]
    for i in range(len(waypoints) - 1):
        punto_inicio = waypoints[i]
        punto_fin = waypoints[i+1]
        
        # Generamos puntos intermedios para X e Y [cite: 133]
        segmento_x = np.linspace(punto_inicio[0], punto_fin[0], puntos_por_segmento)
        segmento_y = np.linspace(punto_inicio[1], punto_fin[1], puntos_por_segmento)
        
        # Usamos .extend() para agregar los puntos a la lista global [cite: 133]
        x_ideal.extend(segmento_x)
        y_ideal.extend(segmento_y)
    
    # Retornamos como arreglos de NumPy [cite: 134]
    return np.array(x_ideal), np.array(y_ideal)

def simular_lidar(n_sectores=36, d_min=0.5, d_max=30.0):
    """
    Simula las lecturas de un sensor RPLIDAR S2.
    """
    # 1. Generar ángulos de 0 a 360 grados [cite: 142]
    angulos_deg = np.linspace(0, 360, n_sectores)
    
    # 2. Distancias aleatorias uniformes [cite: 143]
    distancias = np.random.uniform(d_min, d_max, n_sectores)
    
    # 3. Simulación de obstáculo cercano (índices 5 al 8) [cite: 144]
    distancias[5:9] = np.random.uniform(0.5, 2.0, size=4)
    
    # 4. Normalización (fórmula d_norm = (d - d_min) / (d_max - d_min)) [cite: 145, 146]
    distancias_norm = (distancias - d_min) / (d_max - d_min)
    
    return angulos_deg, distancias, distancias_norm 