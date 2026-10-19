import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ALGORITMO IMPLEMENTADO: PSO

# OBJETIVOS (dos puntos cercanos)
# Define dos puntos en el espacio cartesiano
# El proyecto pide 1, pero por 20pts extra dos
Xd1, Yd1, Zd1 = 0.5, 0.2, 0.8
Xd2, Yd2, Zd2 = 0.52, 0.21, 0.79


# CINEMÁTICA DEL ROBOT
# ✅ Cinemática directa usando matrices DH
# Tabla de parámetros Denavit-Hartenberg (D-H)
# Recibe variables articulares q1-5
# Multiplica matrices DH
# Obtiene posición final (X, Y, Z)
# Calcula la posición final
def forward_kinematics(q):
    q1, q2, q3, q4, q5 = q

    T1 = dh(q1, -np.pi/2, 0.5, 0)
    T2 = dh(q2,  np.pi/2, 0.4, 0)
    T3 = dh(0,   0,       0,   q3)
    T4 = dh(q4, -np.pi/2, 0,   0)
    T5 = dh(q5,  np.pi/2, 0.3, 0)

    T = T1 @ T2 @ T3 @ T4 @ T5

    return T[0:3, 3]


# MATRIZ DENAVIT-HARTENBERG
# Implementa la matriz de transforamación homogénea para derivar las
# expresiones que definen la posición de los ejes X, Y, Z del efector final
def dh(theta, alpha, b, d):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), b*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), b*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])



# ROTACIÓN Y CUATERNIONES
# ⭐ 30 pts: Cálculo de orientación con cuaterniones
# Extrae la matriz de rotación del robot
def get_rotation_matrix(q):
    q1, q2, q3, q4, q5 = q

    T1 = dh(q1, -np.pi/2, 0.5, 0)
    T2 = dh(q2,  np.pi/2, 0.4, 0)
    T3 = dh(0,   0,       0,   q3)
    T4 = dh(q4, -np.pi/2, 0,   0)
    T5 = dh(q5,  np.pi/2, 0.3, 0)

    T = T1 @ T2 @ T3 @ T4 @ T5

    return T[:3, :3]

# y la convierte en cuaternión
# ⭐ 30 pts: Conversión de matriz de rotación a cuaternión
def rot_to_quaternion(R):
    qw = np.sqrt(max(0, 1 + np.trace(R))) / 2
    qx = (R[2,1] - R[1,2]) / (4*qw)
    qy = (R[0,2] - R[2,0]) / (4*qw)
    qz = (R[1,0] - R[0,1]) / (4*qw)
    return np.array([qw, qx, qy, qz])

# Un cuaternión es un número hipercomplejo de cuatro partes que
# se usa en orientaciones y rotaciones tridimensionales
q_desired = np.array([1, 0, 0, 0])  # orientación deseada expresada en cuaternión (sin rotación)


# FITNESS COMPLETO
# ⭐ 20 pts: Penalización de continuidad articular
alpha = 0.7
beta = 0.3
lambda_penalty = 0.5

def fitness(q, target, q_old=None):
    # 1. Error de posición
    pos = forward_kinematics(q)
    error_pos = np.linalg.norm(pos - np.array(target)) # Distancia euclidiana

    # 2. Error de orientación
    # ⭐ 30 pts: Error de orientación usando cuaterniones
    # Calcular la orientación actual del robot comparando actual con q_desired
    R = get_rotation_matrix(q)
    q_current = rot_to_quaternion(R)
    error_ori = np.linalg.norm(q_current - q_desired)

    # alpha 70% y beta 30% (importa más posición que orientación)
    total_error = alpha * error_pos + beta * error_ori

    # 3. Penalización continuidad
    # ⭐ 20 pts: ∆q := (qold - qnew)
    if q_old is not None:
        penalty = np.linalg.norm(q - q_old) # ∆q := (qold − qnew) Mide cuánto cambian las articulaciones
        total_error += lambda_penalty * penalty # Si el cambio es grande → castigo

    # ✅ Función objetivo f(q) = 1 / (1 + e)
    return 1 / (1 + total_error) # f(q) = 1 / (1 + e)


# PSO (Particle Swarm Optimization)
# ✅ Algoritmo metaheurístico: PSO
class PSO:
    def __init__(self, target, q_old=None, num_particles=50, max_iter=100):
        self.target = target    # Objetivo cartesiano (Xd, Yd, Zd)
        self.q_old = q_old      # Configuración anterior del robot ⭐ 20 pts: Uso de solución anterior para evitar saltos bruscos
        self.dim = 5            # Dimensiones
        self.num_particles = num_particles  # Número de soluciones simultáneas
        self.max_iter = max_iter            # Cuántas veces evoluciona el algoritmo

        self.bounds = np.array([    # Límites físicos del robot
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [0.1, 1.0],
            [-np.pi, np.pi],
            [-np.pi, np.pi]
        ])

        self.w = 0.5    # Inercia
        self.c1 = 1.5   # Memoria individual
        self.c2 = 1.5   # Influencia global

        # Soluciones iniciales
        self.positions = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (num_particles, self.dim))
        # Movimiento de partículas (soluciones) con w, c1, c2
        self.velocities = np.random.uniform(-1, 1, (num_particles, self.dim))

        # Mejor solución de cada particula individual
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.array([fitness(p, target, q_old) for p in self.positions])

        # Mejor solución global
        self.gbest_position = self.pbest_positions[np.argmax(self.pbest_scores)]
        self.gbest_score = np.max(self.pbest_scores)

        self.history = []

    def optimize(self):
        for iter in range(self.max_iter):

            for i in range(self.num_particles):

                r1, r2 = np.random.rand(), np.random.rand()

                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                    + self.c2 * r2 * (self.gbest_position - self.positions[i])
                )

                # Se mueve por el espacio de soluciones
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.bounds[:,0], self.bounds[:,1])

                # Evaluar nueva posición
                score = fitness(self.positions[i], self.target, self.q_old)

                # Quedarse con la mejor posición
                if score > self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i]

            # Busca el índice del mejor fitness actual entre todas las partículas.
            max_idx = np.argmax(self.pbest_scores)
            # Comparar: Mejor solución actual (esta iteración) vs mejor solución global (histórica)
            if self.pbest_scores[max_idx] > self.gbest_score:
                self.gbest_score = self.pbest_scores[max_idx]
                self.gbest_position = self.pbest_positions[max_idx]

            self.history.append(self.gbest_score)

            print(f"Iter {iter+1}: Fitness = {self.gbest_score:.5f}")

        return self.gbest_position


# VISUALIZACIÓN 3D
def get_joint_positions(q):
    T = np.eye(4)
    positions = [T[:3, 3]]

    transforms = [
        dh(q[0], -np.pi/2, 0.5, 0),
        dh(q[1],  np.pi/2, 0.4, 0),
        dh(0,     0,       0,   q[2]),
        dh(q[3], -np.pi/2, 0,   0),
        dh(q[4],  np.pi/2, 0.3, 0)
    ]

    for Ti in transforms:
        T = T @ Ti
        positions.append(T[:3, 3])

    return np.array(positions)

def plot_robot(q, target):
    points = get_joint_positions(q)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(points[:,0], points[:,1], points[:,2], '-o')
    ax.scatter(*target, marker='x', s=100)

    ax.set_title("Robot alcanzando objetivo")
    plt.show()


# EJECUCIÓN
print("\n--- PRIMER PUNTO ---")
pso1 = PSO(target=[Xd1, Yd1, Zd1])
best_q1 = pso1.optimize()

print("\n--- SEGUNDO PUNTO (CON CONTINUIDAD) ---")
# ⭐ 20 pts: Segundo punto con continuidad articular
pso2 = PSO(target=[Xd2, Yd2, Zd2], q_old=best_q1)
best_q2 = pso2.optimize()


# RESULTADOS
print("\nConfiguración 1:", best_q1)
print("Configuración 2:", best_q2)

print("\nError final punto 2:",
      np.linalg.norm(forward_kinematics(best_q2) - np.array([Xd2, Yd2, Zd2])))


# GRÁFICAS
# ✅ Gráfica de convergencia
plt.plot(pso1.history, label="Punto 1")
plt.plot(pso2.history, label="Punto 2")
plt.legend()
plt.title("Convergencia")
plt.show()

# VISUALIZACIÓN FINAL
# ✅ Visualización 3D del robot
plot_robot(best_q2, [Xd2, Yd2, Zd2])
