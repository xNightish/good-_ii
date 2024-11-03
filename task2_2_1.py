import numpy as np

def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

def get_a(x, w):
    return w[0] + w[1] * x + w[2] * x**2 + w[3] * np.cos(2 * x) + w[4] * np.sin(2 * x)

def L(x, w):
    return (get_a(x, w) - func(x)) ** 2

def s(x):
    return np.array([1, x, x**2, np.cos(2 * x), np.sin(2 * x)])

def dQdw(x, w):
    return 2 * (get_a(x, w) - func(x)) * s(x).T


coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего

Qe = np.mean([L(x, w) for x in coord_x]) # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for i in range(N):
    k = np.random.randint(0, sz-1) # индекс случайного образа
    gradients = dQdw(coord_x[k], w) # вычисление градиента
    w = w - eta * gradients # обновление параметров модели
    Qe = (1 - lm) * Qe + lm * L(coord_x[k], w).mean() # вычисление экспоненциального скользящего среднего
    
Q = np.mean(L(coord_x, w)) # вычисление среднего эмпирического риска

print(Q)
print(w)