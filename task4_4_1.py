import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(0)

# Исходные параметры распределений классов
r1 = 0.3
D1 = 3.0
mean1 = [2, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 5.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# Моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.4, shuffle=True)

# Обучение с линейным ядром
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

# Предсказание
predict = clf.predict(x_test)

# Подсчет точности
acc = accuracy_score(y_test, predict) * 100  # Переводим в проценты

# Подсчет количества неправильных классификаций
conf_matrix = confusion_matrix(y_test, predict)
incorrect_class_1 = conf_matrix[0, 1]  # Неправильные классификации класса -1
incorrect_class_2 = conf_matrix[1, 0]  # Неправильные классификации класса 1

# Построение графика
plt.figure(figsize=(10, 6))

# Создание сетки для предсказания с более широким диапазоном
xx, yy = np.meshgrid(np.linspace(-20, 20, 200), np.linspace(-20, 20, 200))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Заливка областей
plt.contourf(xx, yy, Z, levels=[Z.min(), 0], color='red', alpha=0.3)  # Заливка для класса -1
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='blue', alpha=0.3)  # Заливка для класса 1

# Отображение точек
plt.scatter(x_train[y_train == -1][:, 0], x_train[y_train == -1][:, 1], color='red', label='Класс -1')
plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], color='blue', label='Класс 1')

# Отображение разделяющей линии
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5)

# Отображение точности и количества неправильных классификаций
plt.text(-14, 8, f'Точность: {acc:.2f}%', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.text(-14, 6, f'Неправильные классификации класса -1: {incorrect_class_1}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.text(-14, 4, f'Неправильные классификации класса 1: {incorrect_class_2}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Классификация с использованием SVM с линейным ядром')
plt.legend()
plt.grid()
plt.xlim(-15, 10)  # Устанавливаем пределы по оси X
plt.ylim(-10, 10)  # Устанавливаем пределы по оси Y

plt.savefig('task4_4_1.png')
plt.show()
