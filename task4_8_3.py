import numpy as np
from matplotlib import pyplot as plt

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 3

# Генерация данных
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)
K = 10
X = np.array([[xx**i for i in range(K)] for xx in coord_x])  # обучающая выборка для поиска коэффициентов модели
Y = coord_y

# Обучающая выборка
X_train = X[::2]  # входы
Y_train = Y[::2]  # целевые значения

# Вычисление матрицы Грама
F = X_train.T @ X_train / X_train.shape[0]

# Вычисление собственных векторов и собственных чисел
L, W = np.linalg.eig(F)

# Сортировка собственных векторов по возрастанию
WW = W[np.argsort(L)]

# Вычисление новых признаков G, оставляя только первые 7 признаков
G = np.dot(X, WW.T)  # Преобразование исходных данных
G = G[:, :7]  # Оставляем только первые 7 признаков

# Формирование матрицы XX_train из образов с новыми признаками G
XX_train = G[::2]

# Вычисление вектора параметров w
w_star = np.linalg.inv(XX_train.T @ XX_train) @ XX_train.T @ Y_train

# Восстановление функции f(x) с использованием матрицы G и вектора параметров w
predict = G @ w_star

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(coord_x, predict, color='red', label='Восстановленная функция', linewidth=2, linestyle='--')
plt.plot(coord_x, coord_y, color='green', label='Исходная функция', linewidth=2)
plt.title("Восстановление функции f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.savefig('task4_8_3.png') # сохранение в файл
plt.show()

