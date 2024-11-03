import numpy as np

# Исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * np.cos(3 * x) + 2

# Функция для вычисления усеченного эмпирического риска Qk(w)
def compute_Qk(w, coord_x, coord_y, k, batch_size):
    Qk = 0
    for i in range(k, k + batch_size):
        x_i = np.array([1, coord_x[i], coord_x[i] ** 2, coord_x[i] ** 3])
        y_i = coord_y[i]
        Qk += (np.dot(w, x_i) - y_i) ** 2
    return Qk / batch_size

# Функция для вычисления псевдоградиента
def compute_gradient(w, coord_x, coord_y, k, batch_size):
    grad = np.zeros(len(w))
    for i in range(k, k + batch_size):
        x_i = np.array([1, coord_x[i], coord_x[i] ** 2, coord_x[i] ** 3])
        y_i = coord_y[i]
        error = np.dot(w, x_i) - y_i
        grad += error * x_i
    return (2 / batch_size) * grad

# Начальные значения и параметры для алгоритма SGD с импульсами Нестерова
coord_x = np.arange(-4.0, 6.0, 0.1)  # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x)  # значения функции по оси ординат

sz = len(coord_x)  # количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001])  # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.])  # начальные значения параметров модели
N = 500  # число итераций алгоритма SGD
lm = 0.02  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20  # размер мини-батча (величина K = 20)
gamma = 0.8  # коэффициент гамма для вычисления импульсов Нестерова
v = np.zeros(len(w))  # начальное значение [0, 0, 0, 0]

Qe = 0  # начальное значение среднего эмпирического риска
np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

# Основной цикл SGD
for n in range(N):
    k = np.random.randint(0, sz - batch_size - 1)  # случайный индекс для мини-батча
    Qk = compute_Qk(w, coord_x, coord_y, k, batch_size)  # вычисление Qk
    grad = compute_gradient(w, coord_x, coord_y, k, batch_size)  # вычисление градиента

    # Обновление импульсов Нестерова
    v = gamma * v + (1 - gamma) * eta * grad
    w = w - v  # обновление параметров

    # Обновление скользящего экспоненциального среднего
    Qe = lm * Qk + (1 - lm) * Qe

# Итоговое значение среднего эмпирического риска для обученной модели
Q = compute_Qk(w, coord_x, coord_y, 0, sz)  # вычисляем Q(a, X) для всей выборки


print(w)
print(Q)
print(Qe)
