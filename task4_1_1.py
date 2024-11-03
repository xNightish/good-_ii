from matplotlib import pyplot as plt
import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.02 * np.exp(-x) - 0.2 * np.sin(3 * x) + 0.5 * np.cos(2 * x) - 7

# вектор признаков для аппроксимации
def s(x, n):
    return np.array([x ** i for i in range(n)])

# вычисление градиента
def Ldw(x, w):
    return 2 * (w.T @ s(x, len(w)) - func(x)) * s(x, len(w)).T

# вычисление функции потерь
def L(x, w):
    return (w.T @ s(x, len(w)) - func(x)) ** 2

coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 1e-3, 1e-4, 1e-5, 1e-6]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
Qe = np.mean([L(x, w) for x in coord_x]) # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for i in range(N):
    k = np.random.randint(0, sz-1) # sz - размер выборки (массива coord_x)
    Qe = (1 - lm) * Qe + lm * L(coord_x[k], w) # скользящее среднее
    gradients = Ldw(coord_x[k], w)
    w = w - eta * gradients # обновление параметров модели
    
# Итоговое значение среднего эмпирического риска для обученной модели
Q = np.mean(L(coord_x, w))

# апроксимирующая модель
y_pred = w @ s(coord_x, len(w))

# Печать результатов
print("Параметры модели:", w)
print("Средний эмпирический риск:", Q)
print("Скользящее среднее риска:", Qe)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(coord_x, coord_y, label='Исходная функция', color='blue')
plt.plot(coord_x, y_pred, label='Аппроксимация', color='red', linestyle='--')
plt.title('Аппроксимация функции')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig('task4_1_1.png')
plt.show()


