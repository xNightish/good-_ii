import numpy as np


def func(x):
    return 0.1 * x + 0.1 * x ** 2 - 0.5 * np.sin(2*x) + 1 * np.cos(4*x) + 10


x = np.arange(-3.0, 4.1, 0.1) # значения по оси абсцисс (Ox) с шагом 0,1
y = np.array(func(x)) # значения функции по оси ординат

N = 22  # размер признакового пространства (степень полинома N-1)
lm = 20  # параметр лямбда для L2-регуляризатора
X = np.array([[a ** n for n in range(N)] for a in x])  # матрица входных векторов
IL = lm * np.eye(N, dtype=int)  # матрица lambda*I
IL[0][0] = 0  # первый коэффициент не регуляризуем

X_train = X[::2]  # обучающая выборка (входы)
Y_train = y[::2]  # обучающая выборка (целевые значения)

w = np.linalg.inv(X_train.T @ X_train + IL) @ X_train.T @ Y_train  # вычисление весов

Q = np.mean((X @ w - y) ** 2)  # вычисление ошибки

print("Коэффициенты w:", w)
print("Средний квадратичный показатель Q:", Q)

# Построение графика
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x, func(x), color='red', label='Истинная функция f(x)', linewidth=2)
plt.plot(x, X @ w, color='green', label='Аппроксимация', linewidth=2, linestyle='--')
plt.title('Аппроксимация функции f(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
# plt.savefig('task4_9_4.png')
plt.show()