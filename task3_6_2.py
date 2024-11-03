import numpy as np
import plotly.graph_objects as go

# Исходные параметры распределений двух классов
np.random.seed(0)
mean1 = np.array([1, -2, 0])  
mean2 = np.array([1, 3, 1])    
r = 0.7
D = 2.0
V = [[D, D * r, D * r], [D * r, D, D * r], [D * r, D * r, D]]  # 3x3 ковариационная матрица

# Моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

# Вычисляем векторы математических ожиданий
mm1 = np.mean(x1.T, axis=0)  # Среднее для класса -1
mm2 = np.mean(x2.T, axis=0)  # Среднее для класса 1

# Вычисляем ковариационную матрицу для всего набора данных
data = np.vstack([x1.T - mm1, x2.T - mm2])
cov_matrix = np.cov(data.T)

# Вычисляем обратную ковариационную матрицу
cov_inv = np.linalg.inv(cov_matrix)

# Вычисляем векторы αy
alpha1 = cov_inv @ mm1  # Для класса -1
alpha2 = cov_inv @ mm2  # Для класса 1

# Вычисляем величины βy
beta1 = np.log(0.5) - 0.5 * (mm1.T @ cov_inv @ mm1)  # Для класса -1
beta2 = np.log(0.5) - 0.5 * (mm2.T @ cov_inv @ mm2)  # Для класса 1

# Построение разделяющей гиперплоскости
x_min, x_max = np.min(x1[0]), np.max(x1[0])
y_min, y_max = np.min(x1[1]), np.max(x1[1])
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

# Уравнение гиперплоскости: w^T * x + b = 0
w = cov_inv @ (mm2 - mm1)
b = (beta2 - beta1)

# Создаем значение z для разделяющей гиперплоскости
zz = (-w[0] * xx - w[1] * yy - b) / w[2]

# Создаем интерактивный 3D график с помощью plotly
fig = go.Figure()

# Добавляем точки классов
fig.add_trace(go.Scatter3d(x=x1[0], y=x1[1], z=x1[2], mode='markers', marker=dict(color='blue', size=5), name='Класс 0'))
fig.add_trace(go.Scatter3d(x=x2[0], y=x2[1], z=x2[2], mode='markers', marker=dict(color='red', size=5), name='Класс 1'))

# Добавляем гиперплоскость
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.5, colorscale='Viridis', name='Гиперплоскость'))

# Настройка графика
fig.update_layout(title='Линейный дискриминант Фишера в 3D', scene=dict(
    xaxis_title='Признак 1',
    yaxis_title='Признак 2',
    zaxis_title='Признак 3'
))

# Сохранение графика в HTML файл
fig.write_html("3d_plot.html")

# Показываем график в браузере
fig.show()

