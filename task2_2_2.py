import numpy as np

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5

def a(x, w):
    return w[0] + w[1] * x + w[2] * x**2 + w[3] * x**3

def L(x, w):
    return (a(x, w) - func(x)) ** 2

def s(x):
    return np.array([1, x, x**2, x**3])

def dQdw(x, w):
    return 2 * (a(x, w) - func(x)) * s(x).T

coord_x = np.arange(-4.0, 6.0, 0.1) # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 50 # размер мини-батча (величина K = 50)

Qe =  np.mean([L(x, w) for x in coord_x])# начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел
for i in range(N):
    k = np.random.randint(0, sz-batch_size-1) # sz - размер выборки (массива coord_x)
    gradients = sum(np.zeros_like(w) + [dQdw(x, w) for x in coord_x[k:k+batch_size]]) / batch_size # градиент
    w = w - eta * gradients # обновление параметров модели
    Qe = lm * Qe + (1 - lm) * L(coord_x[k:k+batch_size], w).mean() # скользящее среднее

    
Q = np.mean(L(coord_x, w)) # средний эмпирический риск
print(Q)
print(w)
print(Qe)