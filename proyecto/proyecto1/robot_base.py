
class robot_base:
    def __init__(self, nombre, capacidad_carga=10, bateria=100, pos_x=0, pos_y=0,yaw=0, basura_recolectada=0, step_dt=0.1):
        self.__nombre = nombre
        self.__capacidad_carga = capacidad_carga         #maximo de peso que puede cargar el robot
        self.__bateria = bateria                         #nivel de bateria iniciar en 100%
        self.__pos_x = pos_x                             #posicion en x del robot
        self.__pos_y = pos_y                             #posicion en y del robot
        self.__yaw = yaw                                 #angulo de posicion del robot
        self.__basura_recolectada = basura_recolectada   #inicial 0
        self.__step_dt = step_dt                         #tiempo de cada paso de movimiento del robot

        self.target_x = 5.0                              #posicion objetivo publico en x
        self.target_y = 5.0                              #posicion objetivo publico en y


        def get_nombre(self):
            return self.__nombre

        def get_bateria(self):
            return self.__bateria

        def get_capacidad_carga(self):
            return self.__capacidad_carga

        def get_pos_x(self):
            return self.__pos_x

        def get_pos_y(self):
            return self.__pos_y

        def get_yaw(self):
            return self.__yaw

        def get_basura_recolectada(self):
            return self.__basura_recolectada
            
        def get_step_dt(self):
            return self.__step_dt

        def _actualizar_posicion(self, x, y, yaw):
            self.__pos_x = x
            self.__pos_y = y
            self.__yaw = yaw


        def _reducir_bateria(self, cantidad):                   #uso de ia
            self.__bateria = max(0, self.__bateria - cantidad)  #uso de ia


    def _recolectar_basura(self, cantidad):
        almacenamiento_basura = self.__capacidad_carga - self.__basura_recolectada
        if cantidad > almacenamiento_basura:                                 #limita la basura si llega almaximo
            cantidad = almacenamiento_basura
        self.__almacenamiento_basura = almacenamiento_basura + cantidad      #aumenta la basura recolectada

    @staticmethod
    def calc_dist_to_goal(pos_x, pos_y, target_x, target_y):
        import math
        return math.sqrt((target_x - pos_x)**2 + (target_y - pos_y)**2)

##arreglar desde aca

    @staticmethod
    def calc_yaw_error(pos_x, pos_y, yaw, target_x, target_y):
        import math
        ang_meta = math.atan2(target_y - pos_y, target_x - pos_x)
        err = ang_meta - yaw
        # Normalización al rango [-π, π]
        err_norm = (err + math.pi) % (2 * math.pi) - math.pi
        return err_norm
        
        
        
    def step(self, v, w):
        import math

        if self.__bateria <= 0:
            return 0.0, True  # sin batería

        # Actualizar yaw
        yaw_nuevo = self.__yaw + w * self.__step_dt
        yaw_nuevo = (yaw_nuevo + math.pi) % (2 * math.pi) - math.pi

        # Actualizar posición
        x_nuevo = self.__pos_x + v * math.cos(yaw_nuevo) * self.__step_dt
        y_nuevo = self.__pos_y + v * math.sin(yaw_nuevo) * self.__step_dt

        self._actualizar_pose(x_nuevo, y_nuevo, yaw_nuevo)

        # Calcular distancia y error angular
        dist = robot_base.calc_dist_to_goal(self.__pos_x, self.__pos_y, self.target_x, self.target_y)
        err_ang = robot_base.calc_yaw_error(self.__pos_x, self.__pos_y, self.__yaw, self.target_x, self.target_y)

        # Recompensa
        reward = -dist - abs(err_ang)
        llegamos = False
        if dist < 0.5:
            reward += 100
            llegamos = True

        return reward, llegamos    


        
    def mover(self):
        raise NotImplementedError("Las clases hijas deben implementar mover()")

    def limpiar(self):
        raise NotImplementedError("Las clases hijas deben implementar limpiar()")
