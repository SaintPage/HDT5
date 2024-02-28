# Programa que se utiliza colas para simular el tiempo de corrida de programas de un sistema operativo
#Ángel de Jesús Mérida Jiménez 23661

import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Parámetros de la simulación
NUM_PROCESOS = 25
INTERVALO = 1
RAM_CAPACIDAD = 100
CPU_VELOCIDAD = 3
NUM_CPUS = 2

class Proceso:
    def __init__(self, env, nombre, ram, cpu):
        self.env = env
        self.nombre = nombre
        self.ram = ram
        self.cpu = cpu
        # Esto genera un número aleatorio entre 1 y 10 y lo asigna a la variable de instancia.
        self.memoria = random.randint(1, 10)
        # Esto genera un número aleatorio entre 1 y 10 y lo asigna a la variable de instancia
        self.instrucciones = random.randint(1, 10)
        self.estado = "new"
        self.tiempo_llegada = None  
        self.tiempo_final = None  
        self.action = self.run  

    def run(self):
        #Registra el tiempo de llegada del proceso al sistema 
        self.tiempo_llegada = self.env.now  
        while True:
            #Si el estado del proceso es “new”, el proceso intenta obtener memoria llamando al método
            if self.estado == "new":
                yield self.env.process(self.obtener_memoria())
            # Si el estado del proceso es “ready”, el proceso intenta ejecutar instrucciones llamando al método
            elif self.estado == "ready":
                yield self.env.process(self.ejecutar_instrucciones())
            #Si el estado del proceso es “waiting”, el proceso espera un tiempo aleatorio entre 1 y 21 unidades de tiempo y luego cambia su estado a “ready”.
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

#Esta es una comprensión de lista que genera una lista de procesos.
procesos = [p for p in generar_procesos(env, NUM_PROCESOS, ram, cpu)]
for proceso in procesos:
    #Para cada objeto Proceso, se llama al método action
    env.process(proceso.action())  
# Para iniciar la simulación.     
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
# Rota a 90 grados los procesos generados
plt.xticks(range(len(nombres_procesos)), nombres_procesos, rotation=90)  
plt.xlabel('Procesos')
plt.ylabel('Tiempo')
plt.title('Tiempo por proceso')
plt.show()