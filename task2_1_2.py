import numpy as np

def func(x):
    return 0.1 * x**2 - np.sin(x) + 5.

def get_a(x, w):
    return w[0] + w[1] * x + w[2] * x**2 + w[3] * x**3

def get_Q(x, w):
    return (get_a(x, w) - func(x)) ** 2

def s(x):
    return np.array([1, x, x**2, x**3])

def dQdw(x, w):
    return 2 * (w.T @ s(x) - func(x)) * s(x).T

coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 200 # число итераций градиентного алгоритма

for i in range(N):
    gradients = sum(np.zeros_like(w) + [dQdw(x, w) for x in coord_x])/ sz 
    w = w - eta * gradients

Q = np.mean(get_Q(coord_x, w))

print(Q)
print(w)



