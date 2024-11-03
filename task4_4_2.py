from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def loss(w, x, y):
    M = np.dot(w, x) * y
    return np.exp(-M)

def df(w, x, y):
    M = np.dot(w, x) * y
    return -np.exp(-M) * x * y

np.random.seed(0)

# Исходные параметры распределений двух классов
r1 = 0.4
D1 = 2.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 3.0
mean2 = [2, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# Моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.array([[1, x[0], x[1]] for x in np.hstack([x1, x2]).T])
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.3, shuffle=True)

n_train = len(x_train)  # Размер обучающей выборки
w = np.array([0.0, 0.0, 0.0])  # Начальные весовые коэффициенты
nt = np.array([0.5, 0.01, 0.01])
N = 500  # Число итераций алгоритма SGD
batch_size = 10  # Размер мини-батча (величина K = 10)

# Алгоритм стохастического градиентного спуска
for n in range(N):
    k = np.random.randint(0, n_train - batch_size)
    x_batch = x_train[k:k + batch_size]
    y_batch = y_train[k:k + batch_size]
    
    # Вычисление псевдоградиента
    grad = df(w, x_batch.T, y_batch).mean(axis=1)
    w -= nt * grad  # Обновление весов

# Вычисление отступов для тестовой выборки
mrgs = np.sort(x_test @ w * y_test)

# Вычисление метрики accuracy
predictions = np.sign(x_test @ w) # Предсказания по отступам
acc = np.mean(predictions == y_test)  # Точность предсказания

# Печать результатов
print("Точность (accuracy):", acc)

# Визуализация
plt.figure(figsize=(10, 6))

# График данных
plt.scatter(x_train[y_train == -1][:, 1], x_train[y_train == -1][:, 2], color='red', label='Класс -1', alpha=0.5)
plt.scatter(x_train[y_train == 1][:, 1], x_train[y_train == 1][:, 2], color='blue', label='Класс +1', alpha=0.5)

# Создание сетки для предсказаний
xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))  # Увеличен диапазон
grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
zz = grid @ w
zz = zz.reshape(xx.shape)

# Закрашивание областей предсказания
plt.contourf(xx, yy, zz, levels=[-np.inf, 0, np.inf], colors=['red', 'blue'], alpha=0.3)

# График разделяющей линии
plt.contour(xx, yy, zz, levels=[0], colors='green', linewidths=2)

# Добавление текста с точностью
plt.text(-7, 8, f'Точность: {acc * 100:.2f}%', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Перемещение легенды в нижний правый угол
plt.legend(loc='lower right')

plt.title('Классификация с использованием SGD')
plt.xlabel('Параметр 1')
plt.ylabel('Параметр 2')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.xlim(-7.5, 8)  # Увеличен диапазон по оси x
plt.ylim(-7.5, 10)  # Увеличен диапазон по оси y
plt.savefig('task4_4_2.png')
plt.show()



