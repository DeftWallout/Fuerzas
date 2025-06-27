import numpy as np
import matplotlib.pyplot as plt

K = 9e+9
DELTA_T = 1e-3
L = 10000
N = 10 # Número de partículas


class Particle:
    def __init__(self, pos, vel, charge=1, mass=1):
        self.mass = mass
        self.charge = charge
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.acc = np.array([0.0, 0.0])
        self.trajectory = [self.pos.copy()]

    def update_dynamics(self, force, delta_t):
        self.acc = force / self.mass
        self.vel += self.acc * delta_t
        self.pos += self.vel * delta_t
        self.apply_periodic_boundary_conditions(L)
        self.trajectory.append(self.pos.copy())

    def apply_periodic_boundary_conditions(self, L):
        self.pos = self.pos % L


def calculate_force(a, b):
    r = b.pos - a.pos
    r = r - L * np.round(r / L)  # condición de imagen mínima
    dist = np.linalg.norm(r)
    if dist == 0:
        return np.array([0.0, 0.0])
    f = K * a.charge * b.charge / (dist ** 3)
    return f * r


# Crear partículas con posiciones y velocidades aleatorias
particles = []
np.random.seed(0)  # para reproducibilidad

for _ in range(N):
    pos = np.random.uniform(0, L, 2)
    vel = np.random.uniform(-500, 500, 2)
    particles.append(Particle(pos, vel))

kinetic_energies = []
potential_energies = []
temperatures = []  # Lista para la temperatura

# Simulación
for _ in range(2000):
    forces = [np.array([0.0, 0.0]) for _ in particles]

    for i in range(N):
        for j in range(i + 1, N):
            f_ij = calculate_force(particles[i], particles[j])
            forces[j] += f_ij
            forces[i] -= f_ij  # acción y reacción

    for i in range(N):
        particles[i].update_dynamics(forces[i], DELTA_T)

    # Calcular energía cinética total
    total_ke = 0.0
    for p in particles:
        v2 = np.dot(p.vel, p.vel)
        total_ke += 0.5 * p.mass * v2
    kinetic_energies.append(total_ke)

    # Calcular temperatura adimensional
    temperature = (2 * total_ke) / N
    temperatures.append(temperature)

    # Energía potencial
    total_pe = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = particles[j].pos - particles[i].pos
            r = r - L * np.round(r / L)
            dist = np.linalg.norm(r)
            if dist != 0:
                total_pe += K * particles[i].charge * particles[j].charge / dist
    potential_energies.append(total_pe)
vx = [p.vel[0] for p in particles]
vy = [p.vel[1] for p in particles]
velocidades = np.sqrt(np.array(vx)**2+np.array(vy)**2)

num_bins = 50
plt.figure()
plt.hist(velocidades, bins=num_bins, density=True, color='skyblue', edgecolor='black')
plt.xlabel('Velocidad |v| (m/s)')
plt.ylabel('Distribución normalizada')
plt.title('Distribución de velocidades al final de la simulación')
plt.grid(True)
plt.show()

# Graficar trayectorias
plt.figure(figsize=(8, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan']

for i, p in enumerate(particles):
    traj = np.array(p.trajectory)
    plt.scatter(traj[:, 0], traj[:, 1], s=1, color=colors[i % len(colors)])
    plt.scatter(traj[0, 0], traj[0, 1], s=50, facecolors='none', edgecolors=colors[i % len(colors)], marker='o')

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Trayectorias de {N} partículas con condiciones periódicas")
plt.xlim(0, L)
plt.ylim(0, L)
plt.grid(True)
plt.gca().set_aspect('equal')
plt.legend()
plt.show()

# Gráfica de energías
tiempos = np.arange(len(kinetic_energies)) * DELTA_T

plt.figure()
plt.plot(tiempos, kinetic_energies, label="Energía cinética")
plt.plot(tiempos, potential_energies, label="Energía potencial")
plt.plot(tiempos, np.array(kinetic_energies) + np.array(potential_energies), label="Energía total", linestyle='--')
plt.xlabel("Tiempo (s)")
plt.ylabel("Energía (J)")
plt.title("Energías del sistema")
plt.legend()
plt.grid(True)
plt.show()

# === Comparación Temperatura vs Energía Total ===

# Asegurarse de que temperaturas y energías estén alineadas
energia_total = np.array(kinetic_energies) + np.array(potential_energies)

# --- Gráfica 1: Energía total y temperatura en el tiempo (ejes duales)
fig, ax1 = plt.subplots()

color1 = 'tab:blue'
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('Energía total (J)', color=color1)
ax1.plot(tiempos, energia_total, color=color1, label='Energía total')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()  # Crear segundo eje Y
color2 = 'tab:orange'
ax2.set_ylabel('Temperatura (adimensional)', color=color2)
ax2.plot(tiempos, temperatures, color=color2, label='Temperatura')
ax2.tick_params(axis='y', labelcolor=color2)

fig.suptitle('Comparación: Energía Total vs Temperatura')
fig.tight_layout()
plt.grid(True)
plt.show()

# --- Gráfica 2: Dispersión directa Temperatura vs Energía Total
plt.figure()
plt.scatter(energia_total, temperatures, color='green', s=10, alpha=0.7)
plt.xlabel("Energía total (J)")
plt.ylabel("Temperatura (adimensional)")
plt.title("Temperatura vs Energía Total")
plt.grid(True)
plt.show()

