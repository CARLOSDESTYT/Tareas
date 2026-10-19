import numpy as np
import matplotlib.pyplot as plt

# Differential Evolution (DE) sirve para optimización numérica como problemas con fórmulas directas o métodos clásicos.
# Está basado en evolución biológica/natural de las especies

# FUNCIONES OBJETIVO
# Michalewicz
# Función no convexa con muchos mínimos locales
# Se usa para probar qué tan bien un algoritmo evita quedarse atrapado en soluciones malas
def michalewicz(x, m=10):
    return -sum(np.sin(x[i]) * (np.sin((i+1)*x[i]**2/np.pi))**(2*m) for i in range(len(x)))

# Six-Hump Camel
# Función clásica de optimización con muchos mínimos locales
# Se usa para probar algoritmos en espacios bidimensionales complejos
def six_hump_camel(x):
    x1, x2 = x
    return (4 - 2.1*x1**2 + (x1**4)/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2



# CLASE DE DIFERENTIAL EVOLUTION (DE)
class DE:
    def __init__(self, func, bounds, dim=2, pop_size=30, max_iter=100, F=0.8, CR=0.9):
        self.func = func        # Función objetivo que queremos minimizar
        self.bounds = bounds    # Límites del espacio de búsqueda
        self.dim = dim          # Número de variables del problema (ej: 2D)
        self.pop_size = pop_size    # Tamaño de la población (número de soluciones)
        self.max_iter = max_iter    # Número de iteraciones del algoritmo

        # Parámetros del algoritmo DE:
        self.F = F      # Factor de mutación (controla cuánto cambia una solución)
        self.CR = CR    # Tasa de cruce (probabilidad de heredar genes del mutante)

        # Inicialización
        # Se crea una población aleatoria dentro de los límites definidos
        # Cada individuo es una posible solución del problema
        self.population = np.random.uniform(bounds[:,0], bounds[:,1], (pop_size, dim))

        # Se evalúa el "fitness" (qué tan buena es cada solución)
        self.fitness = np.array([self.func(ind) for ind in self.population])

        # Se busca la mejor solución inicial (mínimo fitness)
        # Guarda el mejor fitness -> Guarda el individuo correspondiente (solución) al mejor fitness -> Guardar el valor númerico del mejor fitness
        self.best_idx = np.argmin(self.fitness)
        self.best = self.population[self.best_idx]
        self.best_score = self.fitness[self.best_idx]

        # Historial para ver cómo mejora el algoritmo con el tiempo
        self.fitness_history = []

        # Guardar poblaciones para visualizar evolución del algoritmo
        self.initial_pop = self.population.copy()   # Estado inicial
        self.mid_pop = None                         # Estado a mitad de evolución
        self.final_pop = None                       # Estado final


    def mutate(self, i):
        # MUTACIÓN
        # Se eligen 3 individuos aleatorios distintos al actual
        # Esto permite explorar el espacio de búsqueda
        idxs = list(range(self.pop_size))
        idxs.remove(i)
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]

        # Estrategia DE/rand/1:
        # Se genera un nuevo vector combinando diferencias entre soluciones
        # Esto ayuda a explorar nuevas zonas del espacio
        mutant = a + self.F * (b - c)

        # Se asegura que el mutante no salga de los límites del problema
        mutant = np.clip(mutant, self.bounds[:,0], self.bounds[:,1])

        return mutant

    def crossover(self, target, mutant):
        # CRUCE (CROSSOVER)
        # Mezcla el individuo actual (target) con el mutante
        # Esto introduce diversidad en la población (trial)

        trial = np.copy(target)

        for j in range(self.dim):
            # Con probabilidad CR, tomamos el valor del mutante
            # Si no, mantenemos el valor original
            if np.random.rand() < self.CR:
                trial[j] = mutant[j]

        return trial

    def optimize(self):
        for iter in range(self.max_iter):

            for i in range(self.pop_size):

                # 1. Generar mutante
                mutant = self.mutate(i)

                # 2. Hacer cruce entre solución actual y mutante
                trial = self.crossover(self.population[i], mutant)

                # 3. Evaluar nueva solución
                f_trial = self.func(trial)

                # 4. Selección:
                # Si la nueva solución es mejor, reemplaza la anterior
                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial

            # ACTUALIZAR MEJOR GLOBAL
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_score:
                self.best_score = self.fitness[best_idx]
                self.best = self.population[best_idx]

            # Guardar progreso del algoritmo
            self.fitness_history.append(self.best_score)

            # Guardar estado intermedio para visualización
            if iter == self.max_iter // 2:
                self.mid_pop = self.population.copy()

            print(f"Iteración {iter+1}: Mejor fitness = {self.best_score}")

        # Guardar estado final de la población
        self.final_pop = self.population.copy()

        return self.best, self.best_score



# VISUALIZACIÓN
def graficar_poblaciones(de, nombre):
    # Muestra cómo evoluciona la población:
    # inicial → intermedia → final
    fig, axs = plt.subplots(1, 3)

    axs[0].scatter(de.initial_pop[:,0], de.initial_pop[:,1])
    axs[0].set_title("Inicial")

    axs[1].scatter(de.mid_pop[:,0], de.mid_pop[:,1])
    axs[1].set_title("Intermedia")

    axs[2].scatter(de.final_pop[:,0], de.final_pop[:,1])
    axs[2].set_title("Final")

    fig.suptitle(f"Población - {nombre}")
    plt.show()



# EJECUCIÓN DEL ALGORITMO
def ejecutar(func, nombre, bounds):
    print(f"\n--- Ejecutando {nombre} ---")

    # Se crea el algoritmo DE con los parámetros definidos
    de = DE(func=func, bounds=bounds, dim=2, pop_size=30, max_iter=100)

    # Se ejecuta la optimización
    best, score = de.optimize()

    print("\nMejor solución:", best)
    print("Mejor fitness:", score)

    # Gráfica de convergencia
    # Muestra cómo mejora el mejor fitness con el tiempo
    plt.figure()
    plt.plot(de.fitness_history)
    plt.title(f"Convergencia - {nombre}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Mejor Fitness")
    plt.show()

    # Visualización de evolución de población
    graficar_poblaciones(de, nombre)



# MAIN
# Michalewicz en dominio [0, π]
bounds_mich = np.array([[0, np.pi], [0, np.pi]])
ejecutar(michalewicz, "Michalewicz", bounds_mich)

# Six-Hump Camel en dominio definido
bounds_camel = np.array([[-3, 3], [-2, 2]])
ejecutar(six_hump_camel, "Six-Hump Camel", bounds_camel)