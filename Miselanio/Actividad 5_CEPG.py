import numpy as np
import matplotlib.pyplot as plt

# FUNCIONES OBJETIVO
# Rosenbrock (mínimo en [1,1,...,1] = 0)
# Sirve para probar qué tan bien un método encuentra el mínimo de una función complicada
def rosenbrock(x):
    return sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


# Michalewicz (mínimo aprox depende de dimensión)
# Sirve para probar qué tan bien un algoritmo evita caer en mínimos locales (optimización global)
# Tiene muchos mínimos locales y controlas que tan pronunciados son los picos con la variable m
def michalewicz(x, m=10):
    return -sum(np.sin(x[i]) * (np.sin((i+1)*x[i]**2/np.pi))**(2*m) for i in range(len(x)))


# CLASE PSO
class PSO:
    def __init__(self, func, dim, bounds, num_particles=30, max_iter=100):
        self.func = func        # Función a optimizar
        self.dim = dim          # Dimensiones (2)
        self.bounds = bounds    # Límites de búsqueda del algoritmo
        self.num_particles = num_particles      # Número de partículas (soluciones)
        self.max_iter = max_iter    # Iteraciones

        # Parámetros PSO
        self.w = 0.5        # Inercia (mantiene movimiento y dirección)
        self.c1 = 1.5       # Aprendizaje individual (Confiar en su mejor posición personal)
        self.c2 = 1.5       # Aprendizaje social (Confiar en la mejor posición global)

        # Inicialización de posiciones y velocidades aleatorias
        self.positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
        self.velocities = np.random.uniform(-1, 1, (num_particles, dim))

        # Memoria de particulas para guardar su mejor posición
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.array([self.func(p) for p in self.positions])

        # Buscar la mejor posición de todas las particulas hasta ahora
        self.gbest_position = self.pbest_positions[np.argmin(self.pbest_scores)]
        self.gbest_score = np.min(self.pbest_scores)

        self.fitness_history = []

        # Guardar poblaciones Inicial/Intermedia/Final
        self.initial_positions = self.positions.copy()
        self.mid_positions = None
        self.final_positions = None

    def optimize(self):
        for iter in range(self.max_iter):

            for i in range(self.num_particles):
                #c1 y c2
                r1, r2 = np.random.rand(), np.random.rand()

                # Actualizar velocidad
                # Combina Inercia, Pbest y Gbest para su nueva velocidad
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                    + self.c2 * r2 * (self.gbest_position - self.positions[i])
                )

                # Actualizar posición
                self.positions[i] += self.velocities[i]

                # Respetar límites
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

                # Evaluar fitness
                score = self.func(self.positions[i])

                # Actualizar pbest
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i]

            # Actualizar gbest
            min_index = np.argmin(self.pbest_scores)
            if self.pbest_scores[min_index] < self.gbest_score:
                self.gbest_score = self.pbest_scores[min_index]
                self.gbest_position = self.pbest_positions[min_index]

            # Guardar historial
            self.fitness_history.append(self.gbest_score)

            # Guardar población intermedia
            if iter == self.max_iter // 2:
                self.mid_positions = self.positions.copy()

            print(f"Iteración {iter+1}: Mejor fitness = {self.gbest_score}")

        # Guardar población final
        self.final_positions = self.positions.copy()

        return self.gbest_position, self.gbest_score


# GRAFICAR POBLACIONES
def graficar_poblaciones(pso, nombre):

    fig, axs = plt.subplots(1, 3)

    axs[0].scatter(pso.initial_positions[:,0], pso.initial_positions[:,1])
    axs[0].set_title("Inicial")

    axs[1].scatter(pso.mid_positions[:,0], pso.mid_positions[:,1])
    axs[1].set_title("Intermedia")

    axs[2].scatter(pso.final_positions[:,0], pso.final_positions[:,1])
    axs[2].set_title("Final")

    fig.suptitle(f"Población - {nombre}")
    plt.show()


# EJECUCIÓN
def ejecutar(func, nombre, bounds):
    print(f"\n--- Ejecutando {nombre} ---")

    pso = PSO(func=func, dim=2, bounds=bounds, num_particles=30, max_iter=100)
    best_pos, best_score = pso.optimize()

    print("\nMejor solución:", best_pos)
    print("Mejor fitness:", best_score)

    # Gráfica de fitness
    plt.figure()
    plt.plot(pso.fitness_history)
    plt.title(f"Convergencia - {nombre}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Mejor Fitness")
    plt.show()

    # Gráficas de población
    graficar_poblaciones(pso, nombre)


# MAIN
# Rosenbrock
ejecutar(rosenbrock, "Rosenbrock", bounds=(-2, 2))

# Michalewicz
ejecutar(michalewicz, "Michalewicz", bounds=(0, np.pi))