import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

np.random.seed(0)

# Исходные параметры распределений двух классов
params = [
    (0.7, 1.0, [1, -2], 500),   # Класс -1
    (-0.5, 2.0, [0, 2], 1000)  # Класс 1
]

# Моделирование обучающей выборки
x_train = np.vstack([np.random.multivariate_normal(mean, [[D, D * r], [D * r, D]], N)
                      for r, D, mean, N in params])
y_train = np.hstack([np.full(N, i) * 2 - 1 for i, (r, D, mean, N) in enumerate(params)])  # Присваиваем классы -1 и 1

# Вычисление математических ожиданий и ковариационных матриц
m_x = np.array([x_train[y_train == i * 2 - 1].mean(axis=0) for i in range(len(params))])
cov_x = np.array([np.cov(x_train[y_train == i * 2 - 1], rowvar=False) for i in range(len(params))])
inv_c = np.linalg.inv(cov_x)
det_c = np.linalg.det(cov_x)

# Параметры для гауссовского байесовского классификатора
Py = np.array([0.5, 0.5])  # Вероятности классов
Lm = np.ones_like(Py)

# Вычисление квадратичной формы
dx = x_train[:, np.newaxis, :] - m_x
quad_form = np.einsum('ijk,jkl->ijl', dx, inv_c)

# Вычисление логарифма вероятностей
log_probs = -0.5 * np.einsum('ijk,ijk->ij', dx, quad_form) \
            - 0.5 * np.log(det_c) + np.log(Py) + np.log(Lm)

# Предсказание классов
predict = np.argmax(log_probs, axis=1) * 2 - 1  # Преобразуем индексы в классы -1 и 1

# Вычисление матрицы ошибок
cm = confusion_matrix(y_train, predict, labels=[-1, 1])
TN, FP, FN, TP = cm.ravel()  # Разделяем значения матрицы на TP, FP, TN, FN

print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')

# Визуализация
plt.figure(figsize=(10, 6))

# График с распределением классов
plt.title('Распределение классов')
plt.xlabel('Паризнак 1')
plt.ylabel('Паризнак 2')
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Вычисление логарифмических вероятностей для сетки
grid_points = np.c_[xx.ravel(), yy.ravel()]
dx_grid = grid_points[:, np.newaxis, :] - m_x
quad_form_grid = np.einsum('ijk,jkl->ijl', dx_grid, inv_c)
log_probs_grid = -0.5 * np.einsum('ijk,ijk->ij', dx_grid, quad_form_grid) \
                 - 0.5 * np.log(det_c) + np.log(Py) + np.log(Lm)

# Определение зон классификации
Z = np.argmax(log_probs_grid, axis=1) * 2 - 1
Z = Z.reshape(xx.shape)

# Закрашивание зон классификации
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Отображение всех точек классов
plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], color='green', label='Класс 1', alpha=0.5)
plt.scatter(x_train[y_train == -1][:, 0], x_train[y_train == -1][:, 1], color='blue', label='Класс -1', alpha=0.5)

# Отображение линии разделения
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

# Таблица путаницы в верхнем правом углу графика с увеличенными размерами
confusion_matrix_values = np.array([[TP, FP], [FN, TN]])
table = plt.table(cellText=confusion_matrix_values, 
                  colLabels=['y = 1', 'y = -1'],
                  rowLabels=['a(x) = 1', 'a(x) = -1'],
                  cellLoc='center', 
                  loc='upper right',
                  bbox=[0.76, 0.75, 0.2, 0.2])  # Увеличены размеры таблицы

# Изменение размера шрифта
table.auto_set_font_size(False)  # Отключаем автоматическую настройку размера шрифта
table.set_fontsize(14)  # Устанавливаем размер шрифта

# Увеличение ширины столбцов
for i in range(len(confusion_matrix_values)):
    table.auto_set_column_width([0, 1])  # Автоматически устанавливаем ширину столбцов


# Перемещение легенды вниз справа
plt.legend(loc='lower right')
plt.grid()
plt.savefig('task4_5_2.png')
plt.show()















