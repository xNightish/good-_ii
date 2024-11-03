import numpy as np
import matplotlib.pyplot as plt

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return -0.5 * x ** 2 + 0.1 * x ** 3 + np.cos(3 * x) + 7


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


coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

N = 5 # сложность модели (полином степени N-1)
lm_l1 = 2.0 # коэффициент лямбда для L1-регуляризатора
sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.zeros(N) # начальные нулевые значения параметров модели
n_iter = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)
lm_l2 = 2.0 # коэффициент лямбда для L2-регуляризатора

Qe = np.mean(loss(w, coord_x.T, coord_y)) # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел


for i in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1)  # случайный индекс для мини-батча
    
    # Создание мини-батча
    batch_x = coord_x[k:k + batch_size]
    batch_y = coord_y[k:k + batch_size]
    
    # вычисление Qk
    Qk = np.mean(loss(w, batch_x.T, batch_y))
    
    # Обновление скользящего экспоненциального среднего
    Qe = lm * Qk + (1 - lm) * Qe
    
    # вычисление градиента
    W = np.where(np.arange(len(w)) == 0, 0, w)
    grad = dL(w, batch_x.T, batch_y).mean(axis=1) + lm_l1 * np.sign(W)

    # Обновление импульсов Нестерова
    v = eta * grad
    w = w - v  # обновление параметров модели

    
    
# Итоговое значение среднего эмпирического риска для обученной модели
Q = loss(w, coord_x.T, coord_y).mean()  # вычисляем Q(a, X) для всей выборки


# Генерация значений для графика
x_plot = np.linspace(-4.0, 6.0, 100)  # точки для построения графика
y_plot = func(x_plot)  # истинные значения функции
y_pred = model(w, x_plot)  # предсказанные значения модели

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='Истинная функция', color='blue')
plt.plot(x_plot, y_pred, label='Апроксимация модели', color='orange')
plt.title('Апроксимация функции полиномом')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()