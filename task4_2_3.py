import numpy as np
from sklearn import svm

def func(x):
    return np.sin(0.5*x) + 0.2 * np.cos(2*x) - 0.1 * np.sin(4 * x) - 2.5


def model(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3 + w[4] * np.cos(x) + w[5] * np.sin(x)


# обучающая выборка
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

x_train = np.array([[x, x**2, x**3, np.cos(x), np.sin(x)] for x in coord_x])
y_train = coord_y

clf = svm.SVR(kernel='linear')
clf.fit(x_train, y_train)

w = np.array([clf.intercept_[0], *clf.coef_[0]])
X = np.hstack((np.ones((x_train.shape[0], 1)), x_train))

Q = ((w @ X.T - y_train) ** 2).mean()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(coord_x, func(coord_x), color='red', label='Истинная функция f(x)', linewidth=2)
plt.plot(coord_x, X @ w, color='blue', label='Аппроксимация', linewidth=2, linestyle='--')
plt.title('Аппроксимация функции f(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig('task4_2_3.png')
plt.show()