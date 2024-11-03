import numpy as np

np.random.seed(0)

# исходные параметры для формирования образов обучающей выборки
r = 0.7
D = 3.0
mean = [3, 7, -2, 4, 6]
n_feature = 5
V = [[D * r ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N = 1000
X = np.random.multivariate_normal(mean, V, N).T

# Вычисление матрицы Грама
F = (1 / N) * X @ X.T

# Вычисление собственных чисел и векторов
L, W = np.linalg.eig(F)

# Вывод результатов
print("Матрица Грама F:", F, "\nСобственные числа L:", L, "\nСобственные векторы W:", W, sep="\n")


