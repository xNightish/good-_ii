import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def func(x):
    return -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4*x)

def df(x):
    return -0.5 + 0.4 * x - 0.03 * x ** 2 - 1.2 * np.cos(4*x)

# Начальные условия для метода импульсов
eta = 0.1
x = -3.5
N = 200
gamma = 0.8
v = 0

# Сохранение значений для графика
x_values = []
y_values = []

# Вычисление точки минимума с помощью метода импульсов
for i in range(N):
    v = gamma * v + (1 - gamma) * eta * df(x)
    x -= v
    
    # Сохраняем значения для графика
    x_values.append(x)
    y_values.append(func(x))

# Нахождение минимума без метода импульсов
result = minimize(func, x0=-3.5)  # Начальное значение для поиска минимума
x_min = result.x[0]
y_min = func(x_min)

# Создание графика
x_range = np.linspace(-4, 4, 400)
y_range = func(x_range)

plt.plot(x_range, y_range, label='Функция', color='green')
plt.plot(x_values, y_values, color='blue', label='Итерации (Метод импульсов)')  # Линия итераций
plt.scatter(x_values[-1], y_values[-1], color='red', label='Минимум (Метод импульсов)', s=50)
plt.scatter(x_min, y_min, color='black', label='Минимум (Без метода импульсов)', s=50, marker='o')
plt.title('Нахождение минимума функции с помощью метода импульсов и без')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.savefig('task4_9_3.png')
plt.show()


