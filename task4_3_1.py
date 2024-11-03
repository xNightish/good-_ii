import numpy as np
from sklearn import svm


def func(x):
    return np.sin(0.5*x) + 0.2 * np.cos(2*x) - 0.1 * np.sin(4 * x) + 3


# обучающая выборка
coord_x = np.expand_dims(np.arange(-4.0, 6.0, 0.1), axis=1)
coord_y = func(coord_x).ravel()

# тренировочная выборка
x_train = coord_x[::3]
y_train = coord_y[::3]

# тренировочная модель
svr = svm.SVR(kernel='rbf')
svr.fit(x_train, y_train)

# предсказание
predict = svr.predict(coord_x)

# качество восстановленной функции
Q = np.mean((predict - coord_y) ** 2)

print(Q)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(coord_x, func(coord_x), color='red', label='Истинная функция f(x)', linewidth=2)
plt.plot(coord_x, predict, color='blue', label='Аппроксимация', linewidth=2, linestyle='--')
plt.title('Аппроксимация функции f(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig('task4_3_1.png')
plt.show()

