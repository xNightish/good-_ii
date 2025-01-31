from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# Определение функции ядра K(r)
def K(r):
    return (1 / (2 * np.pi)) * np.exp(-0.5 * r**2)

# Определение метрики расстояния ρ(x, x_k)
def distance(x, x_k):
    return np.sum(np.abs(x - x_k), axis=1)

# Алгоритм классификации
def classify(x_test, x_train, y_train, h=1):
    predictions = []
    unique_classes = np.unique(y_train)

    for x in x_test:
        distances = distance(x, x_train) / h
        scores = {y: np.sum(K(distances[y_train == y])) for y in unique_classes}
        predictions.append(max(scores, key=scores.get))

    return predictions


# Создание обучающей и тестовой выборок
np.random.seed(0)
n_feature = 2  # число признаков

# Исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 3]
V1 = [[D1 * r1 ** abs(i - j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [1, 1]
V2 = [[D2 * r2 ** abs(i - j) for j in range(n_feature)] for i in range(n_feature)]

r3 = -0.7
D3 = 1.0
mean3 = [-2, -2]
V3 = [[D3 * r3 ** abs(i - j) for j in range(n_feature)] for i in range(n_feature)]

# Моделирование обучающей выборки
N1, N2, N3 = 200, 150, 190
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T
x3 = np.random.multivariate_normal(mean3, V3, N3).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N1), np.ones(N2), np.ones(N3) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.5, shuffle=True)

# Прогнозирование классов для тестовой выборки
predict = classify(x_test, x_train, y_train)

# Вычисление показателя качества Q(a, X)
Q = np.mean(predict != y_test)

# Вывод результатов
print("Предсказания:", predict)
print("Показатель качества Q:", Q)



# Создание сетки для прогнозирования зон классификации
x_min, x_max = data_x[:, 0].min() - 1, data_x[:, 0].max() + 1
y_min, y_max = data_x[:, 1].min() - 1, data_x[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Прогнозирование для каждой точки в сетке
Z = classify(np.c_[xx.ravel(), yy.ravel()], x_train, y_train)
Z = np.array(Z).reshape(xx.shape)

# Визуализация результатов с зонами классификации
plt.figure(figsize=(10, 6))

# Отображение зон классификации
contour = plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(-0.5, len(np.unique(y_train)) + 1, 1), cmap='viridis_r')

# Тестовая выборка
for i, cls in enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test == cls][:, 0], x_test[y_test == cls][:, 1], color=['orange', 'green', 'blue'][i], marker='o', label=f'Класс {cls} (Тест)', alpha=0.7)

# Настройки графика
plt.title('Классификация тестовой выборки с зонами классификации')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid()

plt.savefig('task5_2_1.png')
# Вывод графика
plt.show()


    
