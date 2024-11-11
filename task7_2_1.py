import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Генерация данных
x = np.arange(-3, 3, 0.1).reshape(-1, 1)
y = 2 * np.cos(x) + 0.5 * np.sin(2 * x) - 0.2 * np.sin(4 * x)

# Подготовка для бустинга
S = np.array(y.ravel())
T = 6
max_depth = 3
algs = []

# Обучение ансамбля решающих деревьев
for _ in range(T):
    algs.append(DecisionTreeRegressor(max_depth=max_depth))
    algs[-1].fit(x, S)
    S -= algs[-1].predict(x)

# Прогнозирование с помощью ансамбля
yy = np.sum([algs[i].predict(x) for i in range(T)], axis=0).reshape(-1, 1)

# Вычисление среднеквадратичной ошибки
QT = np.mean((yy - y) ** 2)
print(f"Среднеквадратичная ошибка: {QT}")

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Исходные данные', color='green', linewidth=2, alpha=0.5)
plt.plot(x, yy, label='Предсказание модели', color='red', linestyle='--', linewidth=2)
plt.title('Аппроксимация функции f(x) с помощью ансамбля решающих деревьев')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig('task7_2_1.png')
plt.show()




