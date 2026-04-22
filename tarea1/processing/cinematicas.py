import numpy as np

# (a) calcular_movimiento(x, y, theta, v, omega, dt=0.1)
 
V_MAX     = 0.8   # m/s   velocidad lineal máxima
OMEGA_MAX = 0.6   # rad/s velocidad angular máxima


def calcular_movimiento(x, y, theta, v, omega, dt=0.1):
   
    # Saturación de velocidades según Tabla 1
    v     = float(np.clip(v,     -V_MAX,     V_MAX))
    omega = float(np.clip(omega, -OMEGA_MAX, OMEGA_MAX))

    x_nuevo     = x     + v * np.cos(theta) * dt
    y_nuevo     = y     + v * np.sin(theta) * dt
    theta_nuevo = theta + omega * dt

    return x_nuevo, y_nuevo, theta_nuevo


#(b) distancia_al_objetivo(x, y, x_meta, y_meta)
def distancia_al_objetivo(x, y, x_meta, y_meta):
    
    return float(np.sqrt((x_meta - x) ** 2 + (y_meta - y) ** 2))

#(c) calcular_error_seguimiento(x_real, y_real, x_ideal, y_ideal)
def calcular_error_seguimiento(x_real, y_real, x_ideal, y_ideal):
    
    n = min(len(x_real), len(x_ideal))
    dx = x_real[:n] - x_ideal[:n]
    dy = y_real[:n] - y_ideal[:n]
    return np.sqrt(dx ** 2 + dy ** 2)
