import numpy as np

# Координаты четырех точек
x = np.array([0, 1, 2, 3])
y = np.array([0.5, 0.8, 0.6, 0.2])

# Множество точек для промежуточного восстановления функции
x_est = np.arange(0, 3.1, 0.1)

# Метрика расстояния
p = lambda x, x_k: np.abs(x_k - x)

# Треугольное окно Парзена
K = lambda r: np.maximum(0, 1 - np.abs(r))  # Треугольное окно

# Вычисление восстановленной функции
h = 1

# Векторизованное вычисление весов
weights = K(p(x_est[:, np.newaxis], x) / h)  # Формируем массив весов

# Рассчитываем восстановленные значения функции
y_est = np.sum(weights * y, axis=1) / np.sum(weights, axis=1)

# Результаты
print("Восстановленные значения функции:", y_est, sep="\n")


# восстановленная функция

import matplotlib.pyplot as plt
plt.plot(x_est, y_est, color='red', label='Восстановленная функция')
plt.scatter(x, y, marker='o', color='blue', label='Исходные данные')
plt.title('Восстановление функции')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()