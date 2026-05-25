# ==========================================
# robot_dron.py
# ==========================================
import random
from robot_base import RobotBase


class RobotDron(RobotBase):
    """
    Robot dron aéreo.
    Hereda de RobotBase. Movimiento rápido pero baja capacidad de carga.
    Debe estar en vuelo para moverse y limpiar.
    """

    def __init__(self, nombre: str, altura_maxima: float):
        # Llamar al constructor padre con capacidad de carga de 5.0 kg
        super().__init__(nombre, capacidad_carga=5.0)
        self.altura_maxima = altura_maxima
        self.en_vuelo = False

    def despegar(self):
        """Activa el modo de vuelo del dron."""
        print(f"[{self.get_nombre()}] Despegando hasta {self.altura_maxima} metros de altura.")
        self.en_vuelo = True

    def mover(self):
        """
        Si está en vuelo, se mueve rápido.
        Si no, retorna sin moverse.
        """
        if self.en_vuelo:
            return self.step(v=2.5, w=1.0)
        return 0.0, False

    def limpiar(self):
        """Solo limpia si está en vuelo. Consume 3.0 de batería y recoge entre 0.1 y 0.4 kg."""
        if self.en_vuelo:
            self._reducir_bateria(3.0)
            basura = random.uniform(0.1, 0.4)
            self._recolectar_basura(basura)
