import numpy as np
import matplotlib.pyplot as plt

# Сигмоидная функция потерь
def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))

# Производная сигмоидной функции потерь по вектору w
def df(w, x, y):
    M = np.dot(w, x) * y
    return -2 * np.exp(M) * x * y / (1 + np.exp(M)) ** 2

data_x = [(5.8, 1.2), (5.6, 1.5), (6.5, 1.5), (6.1, 1.3), (6.4, 1.3), 
           (7.7, 2.0), (6.0, 1.8), (5.6, 1.3), (6.0, 1.6), (5.8, 1.9), 
           (5.7, 2.0), (6.3, 1.5), (6.2, 1.8), (7.7, 2.3), (5.8, 1.2), 
           (6.3, 1.8), (6.0, 1.0), (6.2, 1.3), (5.7, 1.3), (6.3, 1.9), 
           (6.7, 2.5), (5.5, 1.2), (4.9, 1.0), (6.1, 1.4), (6.0, 1.6), 
           (7.2, 2.5), (7.3, 1.8), (6.6, 1.4), (5.6, 2.0), (5.5, 1.0), 
           (6.4, 2.2), (5.6, 1.3), (6.6, 1.3), (6.9, 2.1), (6.8, 2.1), 
           (5.7, 1.3), (7.0, 1.4), (6.1, 1.4), (6.1, 1.8), (6.7, 1.7), 
           (6.0, 1.5), (6.5, 1.8), (6.4, 1.5), (6.9, 1.5), (5.6, 1.3), 
           (6.7, 1.4), (5.8, 1.9), (6.3, 1.3), (6.7, 2.1), (6.2, 2.3), 
           (6.3, 2.4), (6.7, 1.8), (6.4, 2.3), (6.2, 1.5), (6.1, 1.4), 
           (7.1, 2.1), (5.7, 1.0), (6.8, 1.4), (6.8, 2.3), (5.1, 1.1), 
           (4.9, 1.7), (5.9, 1.8), (7.4, 1.9), (6.5, 2.0), (6.7, 1.5), 
           (6.5, 2.0), (5.8, 1.0), (6.4, 2.1), (7.6, 2.1), (5.8, 2.4), 
           (7.7, 2.2), (6.3, 1.5), (5.0, 1.0), (6.3, 1.6), (7.7, 2.3), 
           (6.4, 1.9), (6.5, 2.2), (5.7, 1.2), (6.9, 2.3), (5.7, 1.3), 
           (6.1, 1.2), (5.4, 1.5), (5.2, 1.4), (6.7, 2.3), (7.9, 2.0), 
           (5.6, 1.1), (7.2, 1.8), (5.5, 1.3), (7.2, 1.6), (6.3, 2.5), 
           (6.3, 1.8), (6.7, 2.4), (5.0, 1.0), (6.4, 1.8), (6.9, 2.3), 
           (5.5, 1.3), (5.5, 1.1), (5.9, 1.5), (6.0, 1.5), (5.9, 1.8)]
data_y = [-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 
           -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 
           -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 
           1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 
           1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 
           1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 
           1, -1, -1, -1, -1, 1]

x_train = np.array([[1, x[0], x[1]] for x in data_x])
y_train = np.array(data_y)

n_train = len(x_train)  # размер обучающей выборки
w = np.zeros(3)  # начальные весовые коэффициенты
nt = np.array([1, 0.1, 0.1])  # шаг обучения для каждого параметра w0, w1, w2
lm = 0.01  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
N = 500  # число итераций алгоритма SGD
batch_size = 10  # размер мини-батча (величина K = 10)

loss_history = []  # для хранения значений потерь
Qe = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)])  # начальное значение среднего эмпирического риска
np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

for n in range(N):
    k = np.random.randint(0, n_train - batch_size)  # случайный выбор мини-батча
    x_batch = x_train[k:k + batch_size]
    y_batch = y_train[k:k + batch_size]
    Qk = np.mean([loss(w, x, y) for x, y in zip(x_batch, y_batch)], axis=0)
    w -= nt * np.mean([df(w, x, y) for x, y in zip(x_batch, y_batch)], axis=0)
    Qe = lm * Qk + (1 - lm) * Qe
    loss_history.append(Qk)  # сохраняем значение потерь

# Оценка качества модели
predictions = np.sign(x_train @ w.T)  # предсказания модели
misclassified = predictions != y_train  # неправильно классифицированные точки

# Построение графика потерь
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Потери', color='blue')
plt.title('История потерь')
plt.xlabel('Итерация')
plt.ylabel('Потеря')
plt.grid()
plt.legend()

# Визуализация данных и предсказаний модели
plt.subplot(1, 2, 2)
plt.scatter(x_train[y_train == 1][:, 1], x_train[y_train == 1][:, 2], color='green', label='Класс 1')
plt.scatter(x_train[y_train == -1][:, 1], x_train[y_train == -1][:, 2], color='red', label='Класс -1')


# Построение разделяющей линии
x_values = np.linspace(4.5, 8.5, 100)  # диапазон для x1
y_values = - (w[0] + w[1] * x_values) / w[2]  # вычисляем y для разделяющей линии
plt.plot(x_values, y_values, color='blue', label='Разделяющая линия')

plt.title('Классификация')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('task4_1_2.png')
plt.show()


