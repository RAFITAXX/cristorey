V = int(input("Voltaje es: "))
I = int(input("Corriente es: "))

Resistencia = V/I 
Potencia = V*I 
print(Resistencia,  Potencia)

if Potencia > 1000: print("¡Peligro! Alta disipación de potencia detectada")
else: print("Operación en rangos seguros")

