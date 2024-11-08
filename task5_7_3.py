import numpy as np

# Исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

# Функция модели a(x)
def a(x, w):
    return w[0] + w[1] * x + w[2] * x**2 + w[3] * np.cos(2 * x) + w[4] * np.sin(2 * x)

# Функция потерь L(w)
def loss(w, x, y):
    return (a(x, w) - y) ** 2

# Производная функции потерь по w
def dLdw(w, x, y):
    xi = np.array([1, x, x**2, np.cos(2 * x), np.sin(2 * x)])
    return 2 * (a(x, w) - y) * xi 

coord_x = np.arange(-5.0, 5.0, 0.1)  # Значения по оси абсцисс
coord_y = func(coord_x)  # Значения функции по оси ординат
sz = len(coord_x)  # Количество значений функций (точек)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01])  # Шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.])  # Начальные значения параметров модели
N = 500  # Число итераций алгоритма SGD
lm = 0.02  # Параметр лямбда для вычисления скользящего экспоненциального среднего
# Начальное значение среднего эмпирического риска
Qe = loss(w, coord_x, coord_y).mean()  # Инициализируем Qe первым значением потерь
np.random.seed(0)  # Генерация одинаковых последовательностей псевдослучайных чисел

for _ in range(N):
    k = np.random.randint(0, sz-1)
    # Обновление весов
    grad = dLdw(w, coord_x[k], coord_y[k])  # Вычисляем градиент
    w -= eta * grad 
    # Обновление Qe
    Lk = loss(w, coord_x[k], coord_y[k]) # Вычисляем потери для текущего образа
    Qe = lm * Lk + (1 - lm) * Qe
# Итоговое значение среднего эмпирического риска для обученной модели
Q = np.mean(loss(w, coord_x, coord_y))  # Среднее значение потерь

# Вывод результатов
print("Обученные параметры w:", w)
print("Итоговое значение среднего эмпирического риска Q:", Q)
print("Последнее значение Qe:", Qe)

# визуализация результатов
import matplotlib.pyplot as plt
plt.plot(coord_x, func(coord_x), color='blue', label='Исходная функция f(x)')
plt.plot(coord_x, a(coord_x, w), color='red', label='Аппроксимация', linestyle='--', linewidth=2)
plt.title('Аппроксимация функции f(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig('task5_7_3.png')
plt.show()
