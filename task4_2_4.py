import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


np.random.seed(0)
# исходные параметры распределений классов
r1 = 0.6
D1 = 3.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.7
D2 = 2.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

r3 = 0.5
D3 = 1.0
mean3 = [1, 2]
V3 = [[D3, D3 * r3], [D3 * r3, D3]]

# моделирование обучающей выборки
N = 500
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T
x3 = np.random.multivariate_normal(mean3, V3, N).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

w1, w2, w3 = [[clf.intercept_[i], *clf.coef_[i]] for i in range(3)]


predict = clf.predict(x_test)
Q = sum(predict != y_test)

print(f"Количество неправильных предсказаний: {Q}")
print(f"w1 = {w1}, w2 = {w2}, w3 = {w3}")

import matplotlib.pyplot as plt

# Создаем фигуру и ось
fig, ax = plt.subplots()

plt.title("Классификация")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")


# Отображаем тестовую
scatter1 = ax.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], 
    c='blue', label='Класс 0', alpha=0.6)
scatter2 = ax.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], 
    c='red', label='Класс 1', alpha=0.6)
scatter3 = ax.scatter(x_test[y_test == 2, 0], x_test[y_test == 2, 1], 
    c='green', label='Класс 2', alpha=0.6)

# Завкрасим области классификации
x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1 
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')



# Добавляем легенду
ax.legend()
plt.savefig('task4_2_4.png')

plt.show()



