from matplotlib import pyplot as plt
import numpy as np


def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.1 * x ** 3


def df(x):
    return 0.5 + 0.4 * x - 0.3 * x * x


coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс
coord_y = func(coord_x) # значения по оси ординат (значения функции)

eta = 0.01
x = -4
N = 200

# вычисление точки минимума
for i in range(N):
    x -= eta * df(x)

print('Точка минимума:', x)