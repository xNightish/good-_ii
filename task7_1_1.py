from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


np.random.seed(0)
n_feature = 2

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 7]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [4, 2]
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N1, N2 = 1000, 1200
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)
T = 10
max_depth = 3
b_t = DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
w = np.ones(len(x_train)) / len(x_train) 
algs, alpha = [], []

for t in range(T):
    algs.append(DecisionTreeClassifier(criterion='gini', max_depth=max_depth))
    algs[t].fit(x_train, y_train, sample_weight=w)
    
    predict = algs[t].predict(x_train)
    N = np.sum((y_train != predict) * w)
    alpha.append(0.5 * np.log((1 - N) / N))
    w = w * np.exp(-alpha[t] * y_train * predict)
    w /= np.sum(w)

predict = np.sign(np.sum([alpha[t] * algs[t].predict(x_test) for t in range(T)], axis=0))

Q = np.sum(np.abs(y_test - predict) / 2)

print(f"Количество неправильных предсказаний: {Q} по композиции {T} решающих деревьев")


# Создание сетки для визуализации зон классификации
x_min, x_max = data_x[:, 0].min() - 1, data_x[:, 0].max() + 1
y_min, y_max = data_x[:, 1].min() - 1, data_x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Предсказание классов для каждой точки в сетке
Z = alpha[0] * algs[0].predict(np.c_[xx.ravel(), yy.ravel()])
for t in range(1, T):
    Z += alpha[t] * algs[t].predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.sign(Z).reshape(xx.shape)


# Визуализация зон классификации
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Визуализация тестовых данных и предсказаний
plt.scatter(x_test[y_test == -1][:, 0], x_test[y_test == -1][:, 1], color='red', label='Класс -1', alpha=0.5)
plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], color='blue', label='Класс 1', alpha=0.5)

# Отображение неправильных предсказаний
incorrect_predictions = x_test[predict != y_test]
plt.scatter(incorrect_predictions[:, 0], incorrect_predictions[:, 1], edgecolor='black', facecolor='none', label='Неправильные предсказания', s=100)

plt.title('Классификация с использованием решающих деревьев')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.savefig('task7_1_1.png')
plt.show()


