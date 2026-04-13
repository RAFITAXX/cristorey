import numpy as np

def calcular_movimiento(x, y, theta, v, omega, dt=0.1):
    """
    Calcula la nueva pose del robot siguiendo el modelo cinemático diferencial.
    """
    # 1. Saturación: El robot no puede ir más rápido de lo que sus motores permiten [cite: 208]
    # Restricciones de la Tabla 1 del paper: v max = 0.8 m/s, w max = 0.6 rad/s
    v_sat = np.clip(v, -0.8, 0.8)
    omega_sat = np.clip(omega, -0.6, 0.6)
    
    # 2. Ecuaciones de movimiento (Ecuaciones 1-6 del paper) [cite: 205]
    # x_new = x + v * cos(theta) * dt
    # y_new = y + v * sin(theta) * dt
    # theta_new = theta + omega * dt
    x_nuevo = x + v_sat * np.cos(theta) * dt
    y_nuevo = y + v_sat * np.sin(theta) * dt
    theta_nuevo = theta + omega_sat * dt
    
    return x_nuevo, y_nuevo, theta_nuevo

def distancia_al_objetivo(x, y, x_meta, y_meta):
    """
    Retorna la distancia euclidiana entre el robot y el punto meta[cite: 211].
    """
    distancia = np.sqrt((x_meta - x)**2 + (y_meta - y)**2)
    return float(distancia)

def calcular_error_seguimiento(x_real, y_real, x_ideal, y_ideal):
    """
    Compara la trayectoria real vs la ideal punto por punto[cite: 213].
    """
    # Si los arreglos tienen distinto tamaño, usamos el más corto para evitar errores [cite: 215]
    n = min(len(x_real), len(x_ideal))
    
    x_r, y_r = x_real[:n], y_real[:n]
    x_i, y_i = x_ideal[:n], y_ideal[:n]
    
    # Calculamos la distancia euclidiana en cada punto del arreglo
    errores = np.sqrt((x_r - x_i)**2 + (y_r - y_i)**2)
    
    return errores
