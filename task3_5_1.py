from matplotlib import pyplot as plt
import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [1, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

# обучающая выборка для байесовского классификатора (стандартный формат)
x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# вычисление оценок математических ожиданий
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

# вычисление ковариационных матриц
VV1 = np.cov(x1)
VV2 = np.cov(x2)

# параметры для гауссовского байесовского классификатора
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

# Реализация байесовского гауссовского классификатора

def predict_func(x):
    p1 = np.log(L1 * Py1) - 0.5 * np.log(np.linalg.det(VV1)) - 0.5 * np.dot(np.dot(x - mm1, np.linalg.inv(VV1)), (x - mm1).T)
    p2 = np.log(L2 * Py2) - 0.5 * np.log(np.linalg.det(VV2)) - 0.5 * np.dot(np.dot(x - mm2, np.linalg.inv(VV2)), (x - mm2).T)
    return np.unique(y_train)[np.argmax([p1, p2])]
        

predict = np.array([predict_func(x) for x in x_train])

Q = sum(predict != y_train)

print(predict)
print(Q)



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
plt.savefig('task3_5_1.png')

# Отображаем график
plt.show()

