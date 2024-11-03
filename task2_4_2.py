import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 0.4 * x + 0.1 * np.sin(2*x) + 0.2 * np.cos(3*x)

def df(x):
    return 0.4 + 0.2 * np.cos(2*x) - 0.6 * np.sin(3*x)

# Параметры
n = 1  # скорость обучения
x_start = 4  # начальное значение x
N = 500  # количество итераций
y = 0.7  # коэффициент моментума
v = 0  # начальное значение для v

# Списки для хранения значений
x_min_momentum = []
x_min_no_momentum = []


# Градиентный спуск с моментумом
x_momentum = x_start
v = 0
for i in range(N):
    v = v * y + (1 - y) * n * df(x_momentum - v * y)
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
plt.savefig('task2_4_2.png')
plt.show()