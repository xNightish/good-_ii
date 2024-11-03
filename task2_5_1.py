import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 0.1 * x + 0.1 * x ** 2 - 0.5 * np.sin(2*x) + 1 * np.cos(4*x) + 10

# Генерация данных
x = np.arange(-3.0, 4.1, 0.1)  # значения по оси абсцисс (Ox) с шагом 0.1
y = func(x)  # значения функции по оси ординат

N = 22  # размер признакового пространства (степень полинома N-1)
lm = 20  # параметр лямбда для L2-регуляризатора

# Создание матрицы входных векторов
X = np.array([[a ** n for n in range(N)] for a in x])
IL = lm * np.eye(N)  # матрица lambda*I
IL[0][0] = 0  # первый коэффициент не регуляризуем

# Обучающая выборка
X_train = X[::2]  # входы
Y_train = y[::2]  # целевые значения

# Вычисление весов
w = np.linalg.inv(X_train.T @ X_train + IL) @ X_train.T @ Y_train

# Вычисление ошибки
Q = np.mean((X @ w - y) ** 2)

# предсказание на всей выборке
y_pred = X @ w

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Истинная функция', zorder=5)
plt.plot(x, y, 'b-', color='blue', label='Истинная функция', zorder=5)
plt.scatter(x[::2], Y_train, color='red', label='Обучающие данные', zorder=5)
plt.plot(x, y_pred, 'r-', label='Аппроксимация', color='orange', zorder=5)
plt.title('Полиномиальная регрессия с L2-регуляризацией')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()



