import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4*x)

def df(x):
    return -0.5 + 0.4 * x - 0.03 * x ** 2 - 1.2 * np.cos(4*x)

# Параметры
n = 0.1  # скорость обучения
x_start = -3.5  # начальное значение x
N = 200  # количество итераций
y = 0.8  # коэффициент моментума

# Списки для хранения значений
x_min_momentum = []
x_min_no_momentum = []

# Градиентный спуск с моментумом
x_momentum = x_start
v = 0
for i in range(N):
    v = v * y + (1 - y) * n * df(x_momentum)
    x_momentum -= v
    x_min_momentum.append(x_momentum)

# Градиентный спуск без моментума
x_no_momentum = x_start
for i in range(N):
    x_no_momentum -= n * df(x_no_momentum)
    x_min_no_momentum.append(x_no_momentum)

# Создание графика
x_values = np.linspace(-4, 4, 400)
y_values = func(x_values)

plt.figure(figsize=(12, 6))
plt.plot(x_values, y_values, label='Функция', color='blue')
plt.scatter(x_min_momentum[-1], func(x_min_momentum[-1]), color='red', label='Конечная точка (с импульсом)', zorder=5)
plt.scatter(x_min_no_momentum[-1], func(x_min_no_momentum[-1]), color='green', label='Конечная точка (без импульса)', zorder=5)
plt.plot(x_min_momentum, [func(x) for x in x_min_momentum], label='Траектория (с импульсом)', color='orange')
plt.plot(x_min_no_momentum, [func(x) for x in x_min_no_momentum], label='Траектория (без импульса)', color='purple')

plt.title('Градиентный спуск с и без импульса')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.savefig('')
plt.show()
