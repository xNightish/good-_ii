import numpy as np

np.random.seed(0)

# Исходные параметры распределений трех классов
params = [
    (0.7, 3.0, [1, -2]),  # Класс 0
    (0.5, 2.0, [-3, -1]), # Класс 1
    (0.3, 1.0, [1, 2])    # Класс 2
]

# Моделирование обучающей выборки
N = 1000
x_train = np.vstack([
    np.random.multivariate_normal(mean, [[D, D * r], [D * r, D]], N)
    for r, D, mean in params
])
y_train = np.hstack([np.full(N, i) for i in range(len(params))])

# Вычисление математических ожиданий и ковариационных матриц
m_x = np.array([x_train[y_train == i].mean(axis=0) for i in range(len(params))])
cov_x = np.array([np.cov(x_train[y_train == i], rowvar=False) for i in range(len(params))])
inv_c = np.linalg.inv(cov_x)
det_c = np.linalg.det(cov_x)

# Параметры для гауссовского байесовского классификатора
Py = np.array([0.2, 0.5, 0.3])
Lm = np.ones_like(Py)

# Предсказание классов
dx = x_train[:, np.newaxis, :] - m_x 

# Вычисление квадратичной формы
quad_form = np.einsum('ijk,jkl->ijl', dx, inv_c) 

# Вычисление логарифма вероятностей
log_probs = -0.5 * np.einsum('ijk,ijk->ij', dx, quad_form) \
            - 0.5 * np.log(det_c) + np.log(Py) + np.log(Lm)

# Предсказание классов
predict = np.argmax(log_probs, axis=1)

# Количество неправильных предсказаний
Q = np.sum(predict != y_train)

print(f"Количество неправильных предсказаний: {Q}")


















