import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


# модель
def model(w, x):
    xv = np.array([x ** n for n in range(len(w))])
    return w.T @ xv


# функция потерь
def loss(w, x, y):
    return (model(w, x) - y) ** 2


# производная функции потерь
def dL(w, x, y):
    xv = np.array([x ** n for n in range(len(w))])
    return 2 * (model(w, x) - y) * xv


coord_x = np.arange(-4.0, 6.0, 0.1) # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

N = 5 # сложность модели (полином степени N-1)
lm_l2 = 2 # коэффициент лямбда для L2-регуляризатора
sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.zeros(N) # начальные нулевые значения параметров модели
n_iter = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)

Qe = np.mean([loss(w, x, y) for x, y in zip(coord_x, coord_y)]) # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел


for i in range(n_iter):
    k = np.random.randint(0, sz-batch_size-1)
    wt = np.array([0 if i == 0 else w[i] for i in range(len(w))])
    x_batch = coord_x[k:k+batch_size]
    y_batch = coord_y[k:k+batch_size]
    Qe = (1 - lm) * Qe + lm * np.mean([loss(w, x, y) for x, y in zip(x_batch, y_batch)])
    gradients = sum([dL(w, x, y) for x, y in zip(x_batch, y_batch)]) / batch_size + lm_l2 * wt
    w -= eta * gradients
    
Q = np.mean([loss(w, x, y) for x, y in zip(coord_x, coord_y)])

import matplotlib.pyplot as plt

# Генерация значений для построения графика
x_plot = np.linspace(-4, 6, 100)
y_plot = func(x_plot)
y_model = model(w, x_plot)

# Построение графика оригинальной функции и аппроксимации модели
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='Оригинальная функция', color='blue')
plt.plot(x_plot, y_model, label='Аппроксимация модели', color='red', linestyle='--')
plt.scatter(coord_x, coord_y, color='green', label='Точки данных', s=10)
plt.title('Аппроксимация функции с использованием полиномиальной модели')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('task2_5_2.png')
plt.grid()
plt.show()
