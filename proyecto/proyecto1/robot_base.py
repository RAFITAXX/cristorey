
import math
class RobotBase:

    def __init__(self, nombre: str, capacidad_carga: float,
                 x_inicial=0.0, y_inicial=0.0, yaw_inicial=0.0):
        #atributos privadoss
        self.__nombre = nombre
        self.__capacidad_carga = capacidad_carga
        self.__bateria = 100.0
        self.__pos_x = x_inicial
        self.__pos_y = y_inicial
        self.__yaw = yaw_inicial
        self.__basura_recolectada = 0.0
        self.__step_dt = 0.1  # Intervalo de tiempo de paso de la simulación
        #Coordenadas objetivo (atributo publico)
        self.target_x = 5.0
        self.target_y = 5.0

    # Getters (acceso controlado por atributos privados)
    def get_nombre(self):
        return self.__nombre

    def get_bateria(self):
        return self.__bateria

    def get_pos_x(self):
        return self.__pos_x

    def get_pos_y(self):
        return self.__pos_y

    def get_yaw(self):
        return self.__yaw

    def get_basura_recolectada(self):
        return self.__basura_recolectada

    # Modificación interna
    def _actualizar_pose(self, x, y, yaw):
        # Sobreescribe la posición y orientación actual.
        self.__pos_x = x
        self.__pos_y = y
        self.__yaw = yaw

    def _reducir_bateria(self, cantidad):
        #Le resta batería asegurando que esta no baje de 0.
        self.__bateria = max(0.0, self.__bateria - cantidad)

    def _recolectar_basura(self, cantidad):
        # Se agrega basura respetando la capacidad máxima del robot. Solo se suma el espacio disponible restante
        espacio_disponible = self.__capacidad_carga - self.__basura_recolectada
        a_recolectar = min(cantidad, espacio_disponible)
        self.__basura_recolectada += a_recolectar

    # Métodos estáticos
    @staticmethod
    def calc_dist_to_goal(pos_x, pos_y, target_x, target_y):
        #Calcula la distancia entre la posición actual y la meta
        return math.sqrt((target_x - pos_x) ** 2 + (target_y - pos_y) ** 2)

    @staticmethod
    def calc_yaw_error(pos_x, pos_y, yaw, target_x, target_y):
        #Se calcula el error angular hacia el objetivo y se normaliza el resultado al rango [-π, π].
        angulo_meta = math.atan2(target_y - pos_y, target_x - pos_x)
        err = angulo_meta - yaw
        # Normalización de rango [-π, π]
        err_norm = (err + math.pi) % (2 * math.pi) - math.pi
        return err_norm

    # Simulacion cinematica
    def step(self, v, w):
        # Se ejecuta un paso de simulación cinemática. v será velocidad lineal y w la velocidad angular. Retornaara: (recompensa, llegamos)
        dt = self.__step_dt

        # Si la batería se agota, el robot para
        if self.__bateria <= 0:
            return 0.0, True

        # Calcula el nuevo yaw y normaliza a [-π, π]
        nuevo_yaw = self.__yaw + w * dt
        nuevo_yaw = (nuevo_yaw + math.pi) % (2 * math.pi) - math.pi

        # Se calculan las nuevas posiciones
        nuevo_x = self.__pos_x + v * math.cos(nuevo_yaw) * dt
        nuevo_y = self.__pos_y + v * math.sin(nuevo_yaw) * dt

        # Guarado de pose nueva.
        self._actualizar_pose(nuevo_x, nuevo_y, nuevo_yaw)

        # Calculamos la distancia y el error angular con los métodos estáticos
        distancia = RobotBase.calc_dist_to_goal(nuevo_x, nuevo_y, self.target_x, self.target_y)
        error_angular = RobotBase.calc_yaw_error(nuevo_x, nuevo_y, nuevo_yaw, self.target_x, self.target_y)

        # Recompensa: penalizar lejanía y desvío
        reward = -distancia - abs(error_angular)
        # Se verifica si el robot llegó a la meta (distancia < 0.5 metros)
        llegamos = distancia < 0.5
        if llegamos:
            reward += 100

        return reward, llegamos
    # Métodos abstractos (deben implementarse en subclases)
    def mover(self):
        raise NotImplementedError(
            f"La clase hija de '{self.__class__.__name__}' debe implementar el método mover()."
        )

    def limpiar(self):
        raise NotImplementedError(
            f"La clase hija de '{self.__class__.__name__}' debe implementar el método limpiar()."
        )
