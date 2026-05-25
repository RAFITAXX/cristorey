# ==========================================
# robot_oruga.py
# ==========================================
import random
from robot_base import RobotBase


class RobotOruga(RobotBase):
    """
    Robot tipo oruga (tanque).
    Hereda de RobotBase. Movimiento lento pero mayor capacidad de carga.
    """

    def __init__(self, nombre: str, tension_oruga: float):
        # Llamar al constructor padre con capacidad de carga de 50.0 kg
        super().__init__(nombre, capacidad_carga=50.0)
        self.tension_oruga = tension_oruga

    def ajustar_tension(self):
        """Ajusta y muestra la tensión de las orugas."""
        print(f"[{self.get_nombre()}] Ajustando tension de las orugas al {self.tension_oruga} %.")

    def mover(self):
        """Movimiento lento con giro pronunciado."""
        return self.step(v=0.3, w=0.8)

    def limpiar(self):
        """Consume 4.5 de batería y recoge entre 2.0 y 4.0 kg de basura."""
        self._reducir_bateria(4.5)
        basura = random.uniform(2.0, 4.0)
        self._recolectar_basura(basura)
