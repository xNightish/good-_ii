import numpy as np
import matplotlib.pyplot as plt

# Определение функции f(x)
def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


def model(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3


x_train = np.arange(-4.0, 6.0, 0.1) # обучающая выборка
y_train = func(x_train) # целевые выходные значения
# Формирование матрицы признаков
X = np.vstack([np.ones_like(x_train), x_train, x_train**2, x_train**3]).T

# Решение уравнения для нахождения w
w = np.linalg.inv(X.T @ X) @ (X.T @ y_train)
Q = np.mean((X @ w - y_train) ** 2)

# Вывод значений коэффициентов w
print("Коэффициенты w:", w)
print("Средний квадратичный показатель Q:", Q)

# Построение графика
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Данные', alpha=0.5)
plt.plot(x_train, func(x_train), color='red', label='Истинная функция f(x)', linewidth=2)
plt.plot(x_train, X @ w, color='green', label='Аппроксимация', linewidth=2)
plt.title('Аппроксимация функции f(x) линейной моделью')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig('task3_4_2.png')
plt.show()
