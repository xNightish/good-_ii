from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

np.random.seed(0)

# Исходные параметры распределений классов
r1 = -0.2
D1 = 3.0
mean1 = [1, -5]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -2]
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

# Получение вероятностей принадлежности к положительному классу
y_scores = clf.decision_function(x_test)

# Вычисление значений ROC-кривой
range_t = np.arange(5.7, -7.8, -0.1)
FPR = []
TPR = []

# Вычисление ROC-кривой
for t in range_t:
    y_pred = np.where(y_scores > t, 1, -1)
    
    # Вычисление матрицы ошибок
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    FPR.append(fp / (fp + tn))
    TPR.append(tp / (tp + fn))

# Добавляем начальную и конечную точки
FPR = [0] + FPR + [1]  # Начинаем с 0
TPR = [0] + TPR + [0]  # Заканчиваем на 0

# Визуализация ROC-кривой
plt.figure(figsize=(10, 6))
plt.plot(FPR, TPR, color='green', label=f'ROC-кривая (ROC-AUC = {clf.score(x_test, y_test):.2f})')
plt.fill_between(FPR, TPR, alpha=0.2, color='green')  # Закрашиваем площадь под кривой
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Случайный классификатор')
plt.title('ROC-кривая', fontsize=16)
plt.xlabel('FPR', fontsize=14)
plt.ylabel('TPR', fontsize=14)
plt.legend()
plt.grid()
plt.xlim(0, 1)  # Устанавливаем пределы по оси X
plt.ylim(0, 1.01)  # Устанавливаем пределы по оси Y
plt.show()





