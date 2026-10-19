import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# =========================
# 1. Funciones objetivo
# =========================

def ackley(x, y):
    return (
        math.e
        - 20 * math.exp(-20 * math.sqrt((x**2 + y**2) / 2))
        - math.exp((math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)) / 2)
    )

def esfera(x):
    return sum(v**2 for v in x)

# =========================
# 2. Individuos continuos
# =========================

class Individuo:
    def __init__(self, n_vars, rango):
        self.n_vars = n_vars
        self.rango = rango
        self.x = np.random.uniform(rango[0], rango[1], n_vars)
        self.fitness = None

    def evaluar(self, funcion):
        self.fitness = funcion(self.x)
        return self.fitness

    def mutar(self, prob, sigma=0.1):
        for i in range(self.n_vars):
            if random.random() < prob:
                self.x[i] += np.random.normal(0, sigma)
                self.x[i] = np.clip(self.x[i], self.rango[0], self.rango[1])
        self.fitness = None

class IndividuoAckley:
    def __init__(self, rango):
        self.rango = rango
        self.x = random.uniform(rango[0], rango[1])
        self.y = random.uniform(rango[0], rango[1])
        self.fitness = None

    def evaluar(self, funcion):
        self.fitness = funcion(self.x, self.y)
        return self.fitness

    def mutar(self, prob, sigma=0.1):
        if random.random() < prob:
            self.x += random.gauss(0, sigma)
        if random.random() < prob:
            self.y += random.gauss(0, sigma)
        self.x = np.clip(self.x, self.rango[0], self.rango[1])
        self.y = np.clip(self.y, self.rango[0], self.rango[1])
        self.fitness = None

# =========================
# 3. Algoritmo genético continuo
# =========================

def torneo(poblacion, k=3):
    return min(random.sample(poblacion, k), key=lambda ind: ind.fitness)

def algoritmo_genetico(cfg):
    poblacion = []
    for _ in range(cfg['poblacion']):
        ind = cfg['clase'](*cfg['args'])
        ind.evaluar(cfg['funcion'])
        poblacion.append(ind)

    historial = []

    for _ in range(cfg['ciclos']):
        historial.append(copy.deepcopy(poblacion))
        poblacion.sort(key=lambda x: x.fitness)

        nueva = [copy.deepcopy(poblacion[0])]  # elitismo

        while len(nueva) < cfg['poblacion']:
            p1 = torneo(poblacion)
            p2 = torneo(poblacion)

            hijo = copy.deepcopy(p1)
            if isinstance(hijo, Individuo):
                alpha = random.random()
                hijo.x = alpha * p1.x + (1 - alpha) * p2.x
            else:
                hijo.x = (p1.x + p2.x) / 2
                hijo.y = (p1.y + p2.y) / 2

            hijo.mutar(cfg['mutacion'])
            hijo.evaluar(cfg['funcion'])
            nueva.append(hijo)

        poblacion = nueva

    historial.append(copy.deepcopy(poblacion))
    return min(poblacion, key=lambda x: x.fitness), historial

# =========================
# 4. Experimentos
# =========================

experimentos = [
    {
        'id': 'Ackley',
        'funcion': ackley,
        'clase': IndividuoAckley,
        'args': [(-5, 5)],
        'poblacion': 30,
        'ciclos': 60,
        'mutacion': 0.05,
        'rango': (-5, 5)
    },
    {
        'id': 'Esfera_n4',
        'funcion': esfera,
        'clase': Individuo,
        'args': [4, (-5.12, 5.12)],
        'poblacion': 30,
        'ciclos': 60,
        'mutacion': 0.05,
        'rango': (-5.12, 5.12)
    },
    {
        'id': 'Esfera_n6',
        'funcion': esfera,
        'clase': Individuo,
        'args': [6, (-5.12, 5.12)],
        'poblacion': 30,
        'ciclos': 60,
        'mutacion': 0.05,
        'rango': (-5.12, 5.12)
    }
]

resultados = {}

for exp in experimentos:
    mejor, hist = algoritmo_genetico(exp)
    resultados[exp['id']] = {'mejor': mejor, 'hist': hist, 'cfg': exp}

# =========================
# 5. Gráficas
# =========================

for nombre, data in resultados.items():
    hist = data['hist']
    fits = [min(p, key=lambda x: x.fitness).fitness for p in hist]

    plt.figure(figsize=(6, 4))
    plt.plot(fits, 'o-')
    plt.title(f"{nombre} - Convergencia")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()

    if nombre == 'Ackley':
        rango = data['cfg']['rango']
        x = np.linspace(rango[0], rango[1], 100)
        y = np.linspace(rango[0], rango[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(ackley)(X, Y)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, idx, t in zip(
            axes,
            [0, len(hist)//2, len(hist)-1],
            ['Inicio', 'Medio', 'Fin']
        ):
            ax.contourf(X, Y, Z, levels=20, cmap='viridis')
            pob = hist[idx]
            ax.scatter([i.x for i in pob], [i.y for i in pob], c='red', s=20)
            ax.set_title(t)

        plt.suptitle("Evolución población Ackley")
        plt.show()
