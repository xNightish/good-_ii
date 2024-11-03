import numpy as np

# Установим начальное состояние для воспроизводимости
np.random.seed(0)

# Исходные параметры распределений двух классов
mean1 = np.array([1, -2])
mean2 = np.array([-3, -1])
mean3 = np.array([1, 2])

r = 0.5
D = 1.0
V = [[D, D * r], [D * r, D]]

# Моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T
x3 = np.random.multivariate_normal(mean3, V, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

# Вычисляем векторы математических ожиданий
mu1 = np.mean(x1.T, axis=0)
mu2 = np.mean(x2.T, axis=0)
mu3 = np.mean(x3.T, axis=0)

mu = np.array([mu1, mu2, mu3])

# Вычисляем ковариационную матрицу
data = np.vstack([x1.T - mu1, x2.T - mu2, x3.T - mu3])
cov_matrix = np.cov(data.T)

# Вычисляем обратную ковариационную матрицу
cov_inv = np.linalg.inv(cov_matrix)

# Параметры для линейного дискриминанта Фишера
Py = [0.2, 0.4, 0.4]
L = [1, 1, 1]

# Параметры для логистической регрессии
alpha = np.array([cov_inv @ mu[i] for i in range(len(mu))])
beta = np.array([np.log(L[i] * Py[i]) - 0.5 * mu[i].T @ cov_inv @ mu[i] for i in range(len(mu))])

# Функция предсказания
def predict_func(x):
    return np.unique(y_train)[np.argmax([alpha[i] @ x.T + beta[i] for i in range(len(mu))])]

# Получаем предсказания
predict = np.array([predict_func(x) for x in x_train])
Q = sum(predict != y_train)

print(f"Количество неправильных предсказаний: {Q}")


