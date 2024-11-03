import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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
N1 = 1000
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.5, shuffle=True)

# Обучение модели SVM с линейным ядром
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

# Получение коэффициентов весов
w = np.array([clf.intercept_[0], *clf.coef_[0]])

# Предсказание на тестовой выборке
decision_function = clf.decision_function(x_test)

# Установка порога t = 2
t = 2
predictions = np.where(decision_function >= t, 1, -1)

# Вычисление матрицы ошибок
tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[-1, 1]).ravel()

# Вычисление показателей качества
FPR = fp / (fp + tn)
TPR = tp / (tp + fn)

# Вывод показателей качества
print(f"FPR: {FPR:.2f}, TPR: {TPR:.2f}")

# Визуализация тестовой выборки
plt.figure(figsize=(10, 6))

# Задаем сетку для заливки
xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))

# Получаем предсказания для сетки
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# Рисуем тестовую выборку
plt.scatter(x_test[y_test == -1][:, 0], x_test[y_test == -1][:, 1], color='red', label='Тестовый класс -1', alpha=0.5)
plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], color='blue', label='Тестовый класс +1', alpha=0.5)

# Рисуем разделяющую линию
xlim = plt.xlim()
ylim = plt.ylim()

xx_line = np.linspace(xlim[0], xlim[1])
yy_line = - (w[0] + w[1] * xx_line) / w[2]
plt.plot(xx_line, yy_line, color='green', label='Разделяющая линия', linewidth=2, linestyle='dashed')

plt.title('Тестовая выборка с использованием SVM', fontsize=16)
plt.xlabel('x1', fontsize=14)
plt.ylabel('x2', fontsize=14)
plt.legend()
plt.grid()
plt.show()



