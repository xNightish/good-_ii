import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


def minkowski_distance(x_k, x_m, p, weights=None):
    if weights is None:
        weights = np.ones_like(x_k)  # Если веса не заданы, используем единичные веса
    return (np.sum(weights * np.abs(x_k - x_m) ** p)) ** (1/p)


def knn_predict(x_train, y_train, x_test, k, p):
    predictions = []
    for test_point in x_test:
        # Вычисляем расстояния до всех обучающих образцов
        distances = [minkowski_distance(test_point, train_point, p) for train_point in x_train]
        
        # Получаем индексы k ближайших соседей
        k_indices = np.argsort(distances)[:k]
        
        # Получаем классы ближайших соседей
        k_nearest_labels = [y_train[i] for i in k_indices]
        
        # Голосуем за класс
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    
    return np.array(predictions)


np.random.seed(0)
n_feature = 5 # количество признаков

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 7, -2, 4, 6]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [3, 7, -2, 4, 6] + np.array(range(1, n_feature+1)) * 0.5
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r3 = -0.7
D3 = 1.0
mean3 = [3, 7, -2, 4, 6] + np.array(range(1, n_feature+1)) * -0.5
V3 = [[D3 * r3 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N1, N2, N3 = 100, 120, 90
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T
x3 = np.random.multivariate_normal(mean3, V3, N3).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N1), np.ones(N2), np.ones(N3) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

# Количество ближайших соседей
k = 5

# Прогнозируем классы для тестовой выборки
predict = knn_predict(x_train, y_train, x_test, k, 2)

# Вычисляем показатель качества
Q = np.mean(predict != y_test)

# Выводим результаты
print("Прогнозы для тестовой выборки:", predict, sep='\n')
print("Количество неправильных предсказаний:", Q)
