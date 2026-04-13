import numpy as np

def calcular_IAE(errores, dt):
    """
    Integral del Error Absoluto: sum(|e|) * dt [cite: 182]
    """
    return float(np.sum(np.abs(errores)) * dt)

def calcular_ISE(errores, dt):
    """
    Integral del Error Cuadrático: sum(e^2) * dt 
    """
    return float(np.sum(np.power(errores, 2)) * dt)

def calcular_ITAE(errores, dt):
    """
    Integral del Error Absoluto multiplicado por el Tiempo 
    """
    # Generamos el arreglo de tiempo: [0, dt, 2*dt, ...] [cite: 189]
    t = np.arange(len(errores)) * dt
    return float(np.sum(t * np.abs(errores)) * dt)

def calcular_ITSE(errores, dt):
    """
    Integral del Error Cuadrático multiplicado por el Tiempo 
    """
    t = np.arange(len(errores)) * dt
    return float(np.sum(t * np.power(errores, 2)) * dt)

def calcular_todas_las_metricas(errores, dt):
    """
    Calcula los 4 índices y los retorna en un diccionario redondeado a 2 decimales[cite: 191, 192].
    """
    metricas = {
        "ISE": round(calcular_ISE(errores, dt), 2),
        "IAE": round(calcular_IAE(errores, dt), 2),
        "ITSE": round(calcular_ITSE(errores, dt), 2),
        "ITAE": round(calcular_ITAE(errores, dt), 2)
    }
    return metricas

def calcular_mejora(valor_ppo, valor_mask):
    """
    Calcula la reducción porcentual del error entre la base y la nueva política[cite: 197, 198].
    """
    # Fórmula: ((V_base - V_nueva) / V_base) * 100
    mejora = ((valor_ppo - valor_mask) / valor_ppo) * 100
    return round(mejora, 2)
