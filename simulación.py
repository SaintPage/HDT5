
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Parámetros de la simulación
NUM_PROCESOS = 150
INTERVALO = 10
RAM_CAPACIDAD = 100
CPU_VELOCIDAD = 3
NUM_CPUS = 1

class Proceso:
    def __init__(self, env, nombre, ram, cpu):
        self.env = env
        self.nombre = nombre
        self.ram = ram
        self.cpu = cpu
        self.memoria = random.randint(1, 10)
        self.instrucciones = random.randint(1, 10)
        self.estado = "new"
        self.tiempo_llegada = None  # Se inicializa como None
        self.tiempo_final = None  # Se inicializa como None
        self.action = self.run  # Just assign the generator function

    def run(self):
        self.tiempo_llegada = self.env.now  # Se registra el tiempo de llegada
        while True:
            if self.estado == "new":
                yield self.env.process(self.obtener_memoria())
            elif self.estado == "ready":
                yield self.env.process(self.ejecutar_instrucciones())
            elif self.estado == "waiting":
                yield self.env.timeout(random.randint(1, 21))
                self.estado = "ready"
            elif self.estado == "terminated":
                yield self.env.process(self.liberar_memoria())
                self.tiempo_final = self.env.now  # Se registra el tiempo final
                break
            else:
                yield self.env.timeout(1) 

    def obtener_memoria(self):
        yield self.ram.get(self.memoria)
        self.estado = "ready"

    def ejecutar_instrucciones(self):
        with self.cpu.request() as req:
            yield req
            while self.instrucciones > 0:
                yield self.env.timeout(1)
                self.instrucciones -= min(CPU_VELOCIDAD, self.instrucciones)
            self.estado = "terminated"

    def liberar_memoria(self):
        yield self.ram.put(self.memoria)

def generar_procesos(env, num_procesos, ram, cpu):
    for i in range(num_procesos):
        proceso = Proceso(env, f"Proceso{i}", ram, cpu)
        yield proceso

env = simpy.Environment()
ram = simpy.Container(env, init=RAM_CAPACIDAD, capacity=RAM_CAPACIDAD)
cpu = simpy.Resource(env, capacity=NUM_CPUS)

procesos = [p for p in generar_procesos(env, NUM_PROCESOS, ram, cpu)]
for proceso in procesos:
    env.process(proceso.action())  
env.run()

# Calcular el tiempo promedio y la desviación estándar
tiempos = [p.tiempo_final - p.tiempo_llegada for p in procesos if p.tiempo_final is not None]
if tiempos:
    promedio = np.mean(tiempos)
    desviacion = np.std(tiempos)
    print(f"Tiempo promedio: {promedio}, Desviación estándar: {desviacion}")
else:
    print("No se han completado procesos")

#Gráfica:
nombres_procesos = [p.nombre for p in procesos if p.tiempo_final is not None]
plt.bar(range(len(nombres_procesos)), tiempos)
plt.xticks(range(len(nombres_procesos)), nombres_procesos, rotation=90)  # Rota a 90 grados los procesos generados
plt.xlabel('Procesos')
plt.ylabel('Tiempo')
plt.title('Tiempo por proceso')
plt.show()