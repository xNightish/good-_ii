import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(0)

# Исходные параметры распределений классов
r1 = 0.2
D1 = 3.0
mean1 = [2, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# Моделирование обучающей выборки
N1 = 2500
N2 = 1500
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T
beta = 0.5 ** 2

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.4, shuffle=True)

# Моделирование обучающей выборки
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

# Получение коэффициентов весов
w = np.array([clf.intercept_[0], *clf.coef_[0]])

# Предсказание
predict = clf.predict(x_test)

TP = np.sum((predict == 1) & (y_test == 1))
TN = np.sum((predict == -1) & (y_test == -1))
FP = np.sum((predict == 1) & (y_test == -1))
FN = np.sum((predict == -1) & (y_test == 1))

# Показатели качества
precision = TP / (TP + FP) 
recall = TP / (TP + FN)
F = 2 * precision * recall / (precision + recall)
Fb =  (1 + beta) * precision * recall / (beta * precision + recall)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {F:.2f}, F-beta Score: {Fb:.2f}")

# Визуализация
plt.figure(figsize=(10, 6))



# Рисование тестовой выборки
plt.scatter(x_test[y_test == -1][:, 0], x_test[y_test == -1][:, 1], color='blue', label='Класс -1', alpha=0.5)
plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], color='red', label='Класс +1', alpha=0.5)

# Рисование разделяющей линии
xlim = plt.xlim()
ylim = plt.ylim()

xx = np.linspace(xlim[0], xlim[1])
yy = - (w[0] + w[1] * xx) / w[2]
plt.plot(xx, yy, color='green', label='Разделяющая линия', linewidth=2, linestyle='dashed')

plt.title('Классификация с использованием SVM', fontsize=16)
plt.xlabel('x1', fontsize=14)
plt.ylabel('x2', fontsize=14)
plt.legend()
plt.grid()
plt.show()

