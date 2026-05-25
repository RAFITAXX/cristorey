
import random
from robot_base import RobotBase


class RobotTresRuedas(RobotBase):
    # Robot de tres ruedas.
    # Hereda de RobotBase el movimiento moderado y recoleccion ligera.

    def __init__(self, nombre: str, radio_rueda: float):
        # Llama al constructor padre con capacidad de carga de 20.0 kg
        super().__init__(nombre, capacidad_carga=20.0)
        self.radio_rueda = radio_rueda
        self.ruedas_calibradas = False

    def calibrar_giro(self):
        #Se calibra el sistema de giro del triciclo.
        print(f"[{self.get_nombre()}] Calibrando triciclo con ruedas de {self.radio_rueda} cm...")
        self.ruedas_calibradas = True

    def mover(self):
        #Movimiento a velocidad moderada con un ligero giro.
        return self.step(v=0.8, w=0.2)

    def limpiar(self):
        # Consume 2 de bateria y recoge entre 0.5 y 1.5 kg de basura.
        self._reducir_bateria(2.0)
        basura = random.uniform(0.5, 1.5)
        self._recolectar_basura(basura)
