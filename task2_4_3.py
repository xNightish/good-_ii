import numpy as np
import matplotlib.pyplot as plt

def func(x):
    x = np.clip(x, -10, 10)  # Ограничение значений x
    return 2 * x + 0.1 * x ** 3 + 2 * np.cos(3 * x)

def df(x):
    x = np.clip(x, -10, 10)  # Ограничение значений x
    return 2 + 0.3 * x ** 2 - 6 * np.sin(3 * x)

# Параметры
n = 0.01  # Уменьшенная скорость обучения
x_start = 1  # Начальное значение x
N = 200  # Количество итераций
a = 0.8
G = 0  # Начальное значение для G
e = 0.01

# Списки для хранения значений
x_min_momentum = []
x_min_no_momentum = []

# Градиентный спуск с моментумом
x_momentum = x_start
v = 0
for i in range(N):
    G = a * G + (1 - a) * df(x_momentum) ** 2
    x_momentum -= n * df(x_momentum) / (np.sqrt(G) + e)
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

# Отображение конечных точек
plt.scatter(x_min_momentum[-1], func(x_min_momentum[-1]), color='red', label='Конечная точка (с импульсом)', zorder=5)
plt.scatter(x_min_no_momentum[-1], func(x_min_no_momentum[-1]), color='green', label='Конечная точка (без импульса)', zorder=5)

# Отображение траекторий
plt.plot(x_min_momentum, [func(x) for x in x_min_momentum], label='Траектория (с импульсом)', color='orange')
plt.plot(x_min_no_momentum, [func(x) for x in x_min_no_momentum], label='Траектория (без импульса)', color='purple')

# Отображение всех итераций
plt.scatter(x_min_momentum, [func(x) for x in x_min_momentum], color='orange', alpha=0.5)
plt.scatter(x_min_no_momentum, [func(x) for x in x_min_no_momentum], color='purple', alpha=0.5)

plt.title('Градиентный спуск с и без импульса')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.savefig('task2_4_3.png')
plt.show()


    