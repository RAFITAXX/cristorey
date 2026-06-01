


import pandas as pd
import os
from jugadores import Portero, Defensa, Mediocampista, Delantero


pais_elegido = "Francia"

# 11 jugadores titulares  nombre, edad, altura, dorsal
jugadores_titulares = [
    # 1 Portero
    Portero("Mike Maignan", 30, 1.91, 16,
            atajadas_historicas=10, partidos_sin_goles=8),

    # 4 Defensas
     Defensa("William Saliba", 25, 1.92, 17,
            balones_recuperados=12, duelos_ganados=10),
      Defensa("Dayot Upamecano", 27, 1.86, 4,
            balones_recuperados=15, duelos_ganados=5),
    Defensa("Jules Koundé", 27, 1.78, 5,
            balones_recuperados=10, duelos_ganados=9),  
    Defensa("Théo Hernández", 28, 1.84, 19,
            balones_recuperados=8, duelos_ganados=8),

    # 4 Mediocampistas
    Mediocampista("N'Golo Kanté", 35, 1.68, 13,
                  asistencias=35, pases_clave=18),
    Mediocampista("Aurélien Tchouaméni", 26, 1.85, 8,
                  asistencias=18, pases_clave=14),
    Mediocampista("Manu Koné", 25, 1.76, 6,
                  asistencias=2, pases_clave=20),
    Mediocampista("Ousmane Dembélé", 29, 1.78, 7,
                  asistencias=45, pases_clave=19),

    # 2 Delanteros
    Delantero("Kylian Mbappé", 27, 1.78, 10,
              goles_anotados=888, remates_al_arco=31),
    Delantero("Marcus Thuram", 28, 1.92, 9,
              goles_anotados=32, remates_al_arco=14),
]


print("=" * 55)
print("    SIMULADOR DE CAMPEÓN DEL MUNDO – FRANCIA 🇫🇷")
print("=" * 55)

print("\n📋 ACCIONES EN LA CANCHA:")
print("-" * 45)




# Métodos heredados y propios
portero = jugadores_titulares[0]
print(portero.correr())
print(portero.atajar())
print(portero.despejar())

defensa = jugadores_titulares[1]
print(defensa.correr())
print(defensa.marcar())
print(defensa.anticipar())

medio = jugadores_titulares[6]
print(medio.correr())
print(medio.dar_pase())
print(medio.recuperar_balon())

delantero = jugadores_titulares[9]
print(delantero.correr())
print(delantero.patear_al_arco())
print(delantero.desborde())




# Presentaciones
print("\n🙋 PRESENTACIÓN DEL PLANTEL:")
print("-" * 45)
for jugador in jugadores_titulares:
    print(jugador.saludar())




# Polimorfismo: mostrar_rol() para todos los jugadores
print("\n⚽ ROLES DEL EQUIPO (Polimorfismo):")
print("-" * 45)
for jugador in jugadores_titulares:
    print(f"  #{jugador.dorsal:>2} {jugador.nombre:<25} → {jugador.mostrar_rol()}")





# Construir lista de diccionarios
datos = []
for j in jugadores_titulares:
    fila = {
        "Pais":     pais_elegido,
        "Dorsal":   j.dorsal,
        "Nombre":   j.nombre,
        "Edad":     j.edad,
        "Altura_m": j.altura,
        "Posicion": j.mostrar_rol(),
    }

    # Columnas opcionales según el tipo de jugador
    fila["Goles"]               = j.goles_anotados       if isinstance(j, Delantero)      else 0
    fila["Asistencias"]         = j.asistencias          if isinstance(j, Mediocampista)  else 0
    fila["Atajadas"]            = j.atajadas_historicas  if isinstance(j, Portero)        else 0
    fila["Balones_recuperados"] = j.balones_recuperados  if isinstance(j, Defensa)        else 0

    datos.append(fila)

df = pd.DataFrame(datos)





# 1. Tabla completa del equipo
print("\n📊 TABLA COMPLETA DEL EQUIPO:")
print("-" * 45)
print(df.to_string(index=False))

# 2. Edad promedio del equipo
edad_promedio = df["Edad"].mean()
print(f"\n📈 Edad promedio del equipo:  {edad_promedio:.1f} años")

# 3. Altura máxima del equipo
altura_max = df["Altura_m"].max()
jugador_mas_alto = df.loc[df["Altura_m"].idxmax(), "Nombre"]
print(f"📏 Altura máxima:  {altura_max} m  ({jugador_mas_alto})")

# 4. Cantidad de jugadores por posición
print("\n🧮 Jugadores por posición:")
print(df["Posicion"].value_counts().to_string())

# 5. Promedio de edad por posición
print("\n📅 Promedio de edad por posición:")
print(df.groupby("Posicion")["Edad"].mean().round(1).to_string())

# Exportar a CSV
os.makedirs("output", exist_ok=True)
nombre_csv = f"output/titulares_{pais_elegido.lower()}.csv"
df.to_csv(nombre_csv, index=False, encoding="utf-8")
print(f"\n✅ Archivo exportado exitosamente: {nombre_csv}")
print("=" * 55)
