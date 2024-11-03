import numpy as np

data_x = [(1.3, 5.5), (1.5, 4.9), (4.9, 6.3), (1.5, 5.2), (3.5, 5.7), (1.4, 4.6),
          (4.8, 5.9), (4.5, 5.7), (3.7, 5.5), (1.5, 5.3), (4.6, 6.1), (1.6, 4.8), (1.5, 5.0), (4.0, 5.5),
          (1.3, 4.7), (1.4, 5.0), (1.7, 5.1), (1.5, 5.2), (3.9, 5.2), (1.5, 4.6), (4.1, 5.8), (1.9, 5.1),
          (4.0, 5.5), (4.6, 6.6), (4.5, 6.4), (4.5, 6.0), (4.7, 6.1), (1.3, 4.5), (5.1, 6.0), (4.4, 6.6), 
          (4.0, 6.1), (4.5, 6.2), (3.8, 5.5), (1.5, 5.4), (4.9, 6.9), (3.0, 5.1), (4.5, 5.6), (1.4, 4.9), 
          (4.0, 5.8), (5.0, 6.7), (4.4, 5.5), (3.9, 5.6), (1.4, 4.6), (3.3, 4.9), (3.9, 5.8), (4.2, 5.7), 
          (4.4, 6.3), (1.4, 5.1), (1.6, 5.0), (1.5, 5.1), (4.7, 6.3), (3.6, 5.6), (4.4, 6.7), (1.7, 5.4), 
          (1.3, 4.4), (4.1, 5.6), (1.0, 4.6), (4.3, 6.2), (1.4, 4.4), (4.5, 6.0), (4.7, 6.7), (3.3, 5.0), 
          (1.5, 4.9), (3.5, 5.0), (1.6, 4.7), (1.4, 4.9), (1.4, 4.8), (1.3, 5.0), (4.6, 6.5), (4.0, 6.0), 
          (4.7, 6.1), (1.6, 5.0), (1.4, 5.2), (4.7, 7.0), (1.1, 4.3), (1.6, 5.1), (4.3, 6.4), (1.2, 5.8), 
          (1.9, 4.8), (1.4, 4.8), (1.5, 5.1), (4.8, 6.8), (4.1, 5.7), (1.7, 5.7), (1.6, 5.0), (4.2, 5.7),
          (1.6, 4.8), (1.2, 5.0), (1.3, 4.4), (1.7, 5.4), (4.5, 5.4), (4.2, 5.6), (1.5, 5.4), (1.4, 5.5), 
          (1.4, 5.1), (1.5, 5.1), (4.2, 5.9), (1.5, 5.7), (1.4, 5.0), (1.3, 5.4)]

X = np.array([[x[0], x[1], x[0] * x[1]] for x in data_x]) # матрицу X не менять

# Вычисление матрицы Грама
F =  X.T @ X / X.shape[0]

# Вычисление собственных чисел и векторов
L, W = np.linalg.eig(F)

# Формирование матрицы собственных векторов по убыванию значимости
WW = W[np.argsort(-L)]

# Вычисление нового набора признаков
G = X @ WW.T

# Вывод результатов
print('Новые признаки:', G, sep='\n')
