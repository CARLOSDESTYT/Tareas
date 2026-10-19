import numpy as np
import matplotlib.pyplot as plt


# Función Eggholder
# El mínimo global es aproximadamente -959.64
def eggholder(solution):
    x = solution[0]
    y = solution[1]

    term1 = -(y + 47) * np.sin(np.sqrt(abs(y + x/2 + 47)))
    term2 = -x * np.sin(np.sqrt(abs(x - (y + 47))))

    return term1 + term2


# Parámetros del ABC
sn = 30  # Número de fuentes de alimento (tamaño de la población)
dimension = 2  # Dimensión del problema (x, y)
limite_inferior = -512
limite_superior = 512

max_iteraciones = 100
limite = 20  # Límite de intentos antes de convertir en scout bee


# Inicialización de la población
# Generamos soluciones(fuente de alimento) aleatorias dentro del rango permitido
poblacion = np.random.uniform(limite_inferior, limite_superior, (sn, dimension))

# Guardamos poblaciones para graficar
poblacion_inicial = poblacion.copy()
poblacion_intermedia = None
poblacion_final = None

# Evaluamos el fitness de cada solución
fitness = np.array([eggholder(sol) for sol in poblacion])

# Contador de intentos fallidos
intentos = np.zeros(sn)

# Guardamos el mejor fitness de cada generación
mejor_fitness_historial = []


# Función para calcular la probabilidad
# Mientras menor sea el fitness, mejor es la solución
def calcular_probabilidades(fitness):
    # Convertimos fitness a probabilidad
    fitness_norm = 1 / (1 + abs(fitness))
    prob = fitness_norm / np.sum(fitness_norm)

    return prob


# Inicio del algoritmo ABC
for iteracion in range(max_iteraciones):
    # 1- Cada abeja explora una solución cercana
    for i in range(sn):
        # Seleccionamos otra solución diferente
        k = np.random.randint(0, sn)
        while k == i:
            k = np.random.randint(0, sn)

        # Generamos número aleatorio entre -1 y 1
        phi = np.random.uniform(-1, 1, dimension)

        # Creamos nueva solución
        nueva_solucion = poblacion[i] + phi * (poblacion[i] - poblacion[k])

        # Limitamos los valores al rango permitido
        nueva_solucion = np.clip(nueva_solucion, limite_inferior, limite_superior)

        # Evaluamos fitness de la nueva solución
        nuevo_fitness = eggholder(nueva_solucion)

        # Selección greedy (nos quedamos con la mejor)
        if nuevo_fitness < fitness[i]:

            poblacion[i] = nueva_solucion
            fitness[i] = nuevo_fitness
            intentos[i] = 0

        else:
            intentos[i] += 1


    # 2- bejas observadoras
    probabilidades = calcular_probabilidades(fitness)

    for i in range(sn):

        if np.random.rand() < probabilidades[i]:

            k = np.random.randint(0, sn)
            while k == i:
                k = np.random.randint(0, sn)

            phi = np.random.uniform(-1, 1, dimension)

            nueva_solucion = poblacion[i] + phi * (poblacion[i] - poblacion[k])

            nueva_solucion = np.clip(nueva_solucion, limite_inferior, limite_superior)

            nuevo_fitness = eggholder(nueva_solucion)

            if nuevo_fitness < fitness[i]:

                poblacion[i] = nueva_solucion
                fitness[i] = nuevo_fitness
                intentos[i] = 0

            else:
                intentos[i] += 1


    # 3- Abejas exploradoras (Scout bees)
    # Si una solución no mejora después de cierto límite, se reemplaza por una nueva aleatoria
    for i in range(sn):

        if intentos[i] >= limite:

            poblacion[i] = np.random.uniform(limite_inferior, limite_superior, dimension)
            fitness[i] = eggholder(poblacion[i])
            intentos[i] = 0


    # Guardamos el mejor fitness de esta generación
    mejor = np.min(fitness)
    mejor_fitness_historial.append(mejor)

    print(f"Iteración {iteracion+1}: Mejor fitness = {mejor}")

    # Guardamos población intermedia
    if iteracion == max_iteraciones // 2:
        poblacion_intermedia = poblacion.copy()


# Guardamos población final
poblacion_final = poblacion.copy()


# Mejor solución final
indice_mejor = np.argmin(fitness)
mejor_solucion = poblacion[indice_mejor]
mejor_valor = fitness[indice_mejor]

print("\n==============================")
print("Mejor solución encontrada:")
print("x =", mejor_solucion[0])
print("y =", mejor_solucion[1])
print("fitness =", mejor_valor)
print("==============================")


# Gráficas
plt.plot(mejor_fitness_historial)
plt.title("Convergencia del algoritmo ABC")
plt.xlabel("Iteraciones")
plt.ylabel("Mejor fitness")
plt.grid()
plt.show()


plt.figure()
# Población inicial (círculos)
plt.scatter(poblacion_inicial[:,0], poblacion_inicial[:,1], label="Población inicial", marker="o")
# Población intermedia (triángulos)
plt.scatter(poblacion_intermedia[:,0], poblacion_intermedia[:,1], label="Población intermedia", marker="^")
# Población final (cuadrados)
plt.scatter(poblacion_final[:,0], poblacion_final[:,1], label="Población final", marker="s")

# Etiquetas
plt.title("Evolución de la población (Inicial, Intermedia y Final)")
plt.xlabel("x")
plt.ylabel("y")

# Leyenda
plt.legend()
# Cuadrícula
plt.grid()
# Mostrar gráfica
plt.show()