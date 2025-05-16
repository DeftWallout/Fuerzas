from random import random
import matplotlib.pyplot as plt
import numpy as np



class Particula:
    carga = 1.0
    masa = 1.0
    vel = np.array([0,0])
    pos = np.array([0,0])
    acc = np.array([0,0])
    def __init__(self):
        self.carga = 1.0
        self.masa = 1.0
        self.vel = np.array([random(),random()])
        self.pos = np.array([random(), random()])
        self.acc = np.array([random(),random()])

    def set_pos(self,pos):
        self.pos = pos
    def set_carga(self, carga):
        self.carga = carga
    def set_masa(self, masa):
        self.masa = masa
    def set_vel(self,vel):
        self.vel = vel
    def set_acc(self,acc):
        self.acc = acc

K = 9e9
dt = 10e-2
pas = 0
m_pas = 100
def coulomb(first,second):
    r = second.pos - first.pos

    mg = np.sqrt((r[0]**2)+(r[1]**2))
    print('La distancia es: ', mg)
    ru = r/mg
    force= (K*(a.carga*b.carga)/mg**2)*ru
    return  force

def update(first,second):
    fcc = coulomb(first,second)
    acc_first = fcc/first.masa
    acc_second = -fcc/second.masa
    vel_first = a.vel + acc_first*dt
    vel_second = b.vel + acc_second * dt
    pos_first = a.pos + a.vel*dt + (1/2)*acc_first*dt
    pos_second = b.pos + b.vel * dt + (1 / 2) * acc_second * dt
    a.set_pos(pos_first)
    b.set_pos(pos_second)
    a.set_vel(vel_first)
    b.set_vel(vel_second)
    a.set_acc(acc_first)
    b.set_acc(acc_second)

a = Particula()
b = Particula()

trayectoria_a = [a.pos.copy()]
trayectoria_b = [b.pos.copy()]
while pas < m_pas:
    print(f"--- Paso {pas+1} ---")
    print(f"Posición A: {a.pos}, Velocidad A: {a.vel}")
    print(f"Posición B: {b.pos}, Velocidad B: {b.vel}")
    f = coulomb(a, b)
    print(f"Fuerza de Coulomb: {f}")

    # Condición de salida opcional: si las partículas se alejan más de 10 unidades
    distancia = np.linalg.norm(b.pos - a.pos)
    if distancia > 10:
        print("Las partículas están muy lejos. Terminando simulación.")
        break

    update(a, b)
    trayectoria_a.append(a.pos.copy())
    trayectoria_b.append(b.pos.copy())
    pas += 1

trayectoria_a = np.array(trayectoria_a)
trayectoria_b = np.array(trayectoria_b)

# Visualización
plt.figure(figsize=(8, 6))
plt.plot(trayectoria_a[:, 0], trayectoria_a[:, 1], label='Partícula A', color='blue')
plt.plot(trayectoria_b[:, 0], trayectoria_b[:, 1], label='Partícula B', color='red')
plt.scatter(trayectoria_a[0, 0], trayectoria_a[0, 1], color='blue', marker='o', label='Inicio A')
plt.scatter(trayectoria_b[0, 0], trayectoria_b[0, 1], color='red', marker='o', label='Inicio B')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trayectoria de dos partículas cargadas')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()







