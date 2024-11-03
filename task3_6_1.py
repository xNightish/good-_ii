import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = [1, -2]
mean2 = [1, 3]
r = 0.7
D = 2.0
V = [[D, D * r], [D * r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# вычисление оценок МО и ковариационной матрицы
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T])
VV = np.array([[np.dot(a[0], a[0]) / (2*N), np.dot(a[0], a[1]) / (2*(N - 1))],
                [np.dot(a[1], a[0]) / (2*(N - 1)), np.dot(a[1], a[1]) / (2*(N - 1))]])

cov_inv = np.linalg.inv(VV)


# Функция для классификации с использованием линейного дискриминанта Фишера
def fisher_discriminant(x):
    p1 = - 0.5 * (mm1.T @ cov_inv @ mm1) + (x.T @ cov_inv @ mm1)
    p2 = - 0.5 * (mm2.T @ cov_inv @ mm2) + (x.T @ cov_inv @ mm2)
    return np.unique(y_train)[np.argmax([p1, p2])]

# Классификация обучающей выборки
predict = [fisher_discriminant(x) for x in x_train]

# Вычисление показателя качества
Q = sum(predict != y_train)

print(Q)

import matplotlib.pyplot as plt

# Создаем фигуру и ось
fig, ax = plt.subplots()

# Отображаем обучающую выборку
scatter1 = ax.scatter(x_train[y_train == -1, 0], x_train[y_train == -1, 1], 
                       c='blue', label='Класс -1', alpha=0.5)
scatter2 = ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], 
                       c='green', label='Класс 1', alpha=0.5)

# Отображаем неправильно классифицированные точки
incorrectly_classified = [i for i, (p, y) in enumerate(zip(predict, y_train)) if p != y]
ax.scatter(x_train[incorrectly_classified, 0], x_train[incorrectly_classified, 1], 
           c='red', marker='x', s=100, label='Неправильно классифицированные')

# Добавляем заголовок и метки осей
ax.set_title('Байесовский гауссовский классификатор')
ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')

# Добавляем легенду
ax.legend()
plt.savefig('task3_6_1.png')

# Отображаем график
plt.show()