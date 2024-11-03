import numpy as np

# Установка начального значения генератора случайных чисел для воспроизводимости
np.random.seed(0)

n_total = 1000  # число образов выборки
n_features = 200  # число признаков

# Создание разреженной матрицы
table = np.zeros(shape=(n_total, n_features))
for _ in range(100):
    i, j = np.random.randint(0, n_total), np.random.randint(0, n_features)
    table[i, j] = np.random.randint(1, 10)

# Вычисление матрицы Грама
F = table.T @ table / n_total

# Вычисление собственных чисел и векторов
L, W = np.linalg.eig(F)

# Сохранение отсортированных собственных векторов в переменной WW
WW = W[np.argsort(-L)]

# Вычисление нового набора признаков в пространстве векторов WW
data_x = table @ WW

# Удаление признаков с собственными числами < 0.01
data_x = data_x[:, L > 0.01]

# Вывод результатов
print('Новые признаки:', data_x, sep='\n')
