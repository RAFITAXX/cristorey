# jugadores.py
# Contiene la clase padre Jugador y las clases hijas Portero, Defensa, Mediocampista y Delantero.

class Jugador:
    """Clase padre que representa a un jugador genérico de fútbol."""

    def __init__(self, nombre, edad, altura, dorsal):
        self.nombre = nombre
        self.edad = edad        # en años
        self.altura = altura    # en metros
        self.dorsal = dorsal

    def correr(self):
        """Retorna un mensaje indicando que el jugador está corriendo."""
        return f"{self.nombre} está corriendo por la cancha."

    def mostrar_rol(self):
        """Método base de polimorfismo: retorna el rol genérico del jugador."""
        return "Jugador"

    def saludar(self):
        """El jugador se presenta al público."""
        return f"¡Hola! Soy {self.nombre}, llevo el dorsal #{self.dorsal}."

    def estadisticas_base(self):
        """Muestra las estadísticas básicas del jugador."""
        return (f"{self.nombre} | Edad: {self.edad} años | "
                f"Altura: {self.altura} m | Dorsal: #{self.dorsal}")


class Portero(Jugador):
    """Clase hija que representa a un portero."""

    def __init__(self, nombre, edad, altura, dorsal, atajadas_historicas, partidos_sin_goles):
        super().__init__(nombre, edad, altura, dorsal)
        self.atajadas_historicas = atajadas_historicas   # atributo propio 1
        self.partidos_sin_goles = partidos_sin_goles      # atributo propio 2

    def mostrar_rol(self):
        """Polimorfismo: retorna el rol específico de Portero."""
        return "Portero"

    def atajar(self):
        """El portero realiza una atajada."""
        return f"{self.nombre} lanzó un manotazo y atajó el balón."

    def despejar(self):
        """El portero despeja el balón con el puño."""
        return f"{self.nombre} sale del arco y despeja el balón con el puño."


class Defensa(Jugador):
    """Clase hija que representa a un defensa."""

    def __init__(self, nombre, edad, altura, dorsal, balones_recuperados, duelos_ganados):
        super().__init__(nombre, edad, altura, dorsal)
        self.balones_recuperados = balones_recuperados  # atributo propio 1
        self.duelos_ganados = duelos_ganados            # atributo propio 2

    def mostrar_rol(self):
        """Polimorfismo: retorna el rol específico de Defensa."""
        return "Defensa"

    def marcar(self):
        """El defensa marca a un rival."""
        return f"{self.nombre} marca de cerca al delantero rival."

    def anticipar(self):
        """El defensa anticipa un pase en profundidad."""
        return f"{self.nombre} anticipa el pase y corta el ataque rival."


class Mediocampista(Jugador):
    """Clase hija que representa a un mediocampista."""

    def __init__(self, nombre, edad, altura, dorsal, asistencias, pases_clave):
        super().__init__(nombre, edad, altura, dorsal)
        self.asistencias = asistencias   # atributo propio 1
        self.pases_clave = pases_clave   # atributo propio 2

    def mostrar_rol(self):
        """Polimorfismo: retorna el rol específico de Mediocampista."""
        return "Mediocampista"

    def dar_pase(self):
        """El mediocampista realiza un pase."""
        return f"{self.nombre} da un pase preciso al compañero en posición."

    def recuperar_balon(self):
        """El mediocampista recupera el balón en el mediocampo."""
        return f"{self.nombre} presiona alto y recupera el balón en el mediocampo."


class Delantero(Jugador):
    """Clase hija que representa a un delantero."""

    def __init__(self, nombre, edad, altura, dorsal, goles_anotados, remates_al_arco):
        super().__init__(nombre, edad, altura, dorsal)
        self.goles_anotados = goles_anotados       # atributo propio 1
        self.remates_al_arco = remates_al_arco     # atributo propio 2

    def mostrar_rol(self):
        """Polimorfismo: retorna el rol específico de Delantero."""
        return "Delantero"

    def patear_al_arco(self):
        """El delantero remata al arco."""
        return f"{self.nombre} patea fuerte al arco. ¡Goool!"

    def desborde(self):
        """El delantero desborda por la banda."""
        return f"{self.nombre} desborda al defensa por la banda y centra el balón."
