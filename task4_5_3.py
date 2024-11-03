import numpy as np
import matplotlib.pyplot as plt

# Логарифмическая функция потерь
def loss(w, x, y):
    M = np.dot(w, x) * y
    return np.exp(-M)

def df(w, x, y):
    M = np.dot(w, x) * y
    return -np.exp(-M) * x * y / (1 + np.exp(-M)) / np.log(2)

data_x = [(5.8, 1.2), (5.6, 1.5), (6.5, 1.5), (6.1, 1.3), (6.4, 1.3), (7.7, 2.0), (6.0, 1.8), 
           (5.6, 1.3), (6.0, 1.6), (5.8, 1.9), (5.7, 2.0), (6.3, 1.5), (6.2, 1.8), (7.7, 2.3), 
           (5.8, 1.2), (6.3, 1.8), (6.0, 1.0), (6.2, 1.3), (5.7, 1.3), (6.3, 1.9), (6.7, 2.5), 
           (5.5, 1.2), (4.9, 1.0), (6.1, 1.4), (6.0, 1.6), (7.2, 2.5), (7.3, 1.8), (6.6, 1.4), 
           (5.6, 2.0), (5.5, 1.0), (6.4, 2.2), (5.6, 1.3), (6.6, 1.3), (6.9, 2.1), (6.8, 2.1), 
           (5.7, 1.3), (7.0, 1.4), (6.1, 1.4), (6.1, 1.8), (6.7, 1.7), (6.0, 1.5), (6.5, 1.8), 
           (6.4, 1.5), (6.9, 1.5), (5.6, 1.3), (6.7, 1.4), (5.8, 1.9), (6.3, 1.3), (6.7, 2.1), 
           (6.2, 2.3), (6.3, 2.4), (6.7, 1.8), (6.4, 2.3), (6.2, 1.5), (6.1, 1.4), (7.1, 2.1), 
           (5.7, 1.0), (6.8, 1.4), (6.8, 2.3), (5.1, 1.1), (4.9, 1.7), (5.9, 1.8), (7.4, 1.9), 
           (6.5, 2.0), (6.7, 1.5), (6.5, 2.0), (5.8, 1.0), (6.4, 2.1), (7.6, 2.1), (5.8, 2.4), 
           (7.7, 2.2), (6.3, 1.5), (5.0, 1.0), (6.3, 1.6), (7.7, 2.3), (6.4, 1.9), (6.5, 2.2), 
           (5.7, 1.2), (6.9, 2.3), (5.7, 1.3), (6.1, 1.2), (5.4, 1.5), (5.2, 1.4), (6.7, 2.3), 
           (7.9, 2.0), (5.6, 1.1), (7.2, 1.8), (5.5, 1.3), (7.2, 1.6), (6.3, 2.5), (6.3, 1.8), 
           (6.7, 2.4), (5.0, 1.0), (6.4, 1.8), (6.9, 2.3), (5.5, 1.3), (5.5, 1.1), (5.9, 1.5), 
           (6.0, 1.5), (5.9, 1.8)]
data_y = [-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 
           -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 
           -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 
           -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 
           -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1]

x_train = np.array([[1, x[0], x[1]] for x in data_x])
y_train = np.array(data_y)

n_train = len(x_train)  # размер обучающей выборки
w = np.zeros(3)  # начальные весовые коэффициенты
nt = np.array([0.5, 0.01, 0.01])  # шаг обучения для каждого параметра w0, w1, w2
N = 500  # число итераций алгоритма SGD

np.random.seed(0)

for i in range(N):
    k = np.random.randint(0, n_train - 1) 
    w -= nt * df(w, x_train[k], y_train[k])

# Прогнозирование
predict = np.sign(w @ x_train.T)

TP = np.sum((predict == 1) & (y_train == 1))
TN = np.sum((predict == -1) & (y_train == -1))
FP = np.sum((predict == 1) & (y_train == -1))
FN = np.sum((predict == -1) & (y_train == 1))

precision = TP / (TP + FP) 
recall = TP / (TP + FN) 
f1 = 2 * precision * recall / (precision + recall)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
print(f"Weights: {w}")

# Визуализация
plt.figure(figsize=(10, 6))

# Рисование обучающей выборки
plt.scatter(x_train[y_train == -1][:, 1], x_train[y_train == -1][:, 2], color='red', label='Класс -1', alpha=0.5)
plt.scatter(x_train[y_train == 1][:, 1], x_train[y_train == 1][:, 2], color='blue', label='Класс +1', alpha=0.5)

# Закрашивание областей предсказания
xx, yy = np.meshgrid(np.linspace(4.8, 8, 200), np.linspace(0.8, 3, 200))  # Увеличен диапазон
grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
zz = grid @ w
zz = zz.reshape(xx.shape)

plt.contourf(xx, yy, zz, levels=[-np.inf, 0, np.inf], colors=['green', 'blue'], alpha=0.3)

# Рисование разделяющей линии
x = np.linspace(4.8, 8, 100)
y = - (w[0] + w[1] * x) / w[2]
plt.plot(x, y, color='green', label='Разделяющая линия', linewidth=2, linestyle='dashed')

# Добавление precision и recall на график
plt.text(5.0, 2.8, f'Precision: {precision:.2f}', fontsize=12, color='black')
plt.text(5.0, 2.6, f'Recall: {recall:.2f}', fontsize=12, color='black')
plt.text(5.0, 2.4, f'F1 Score: {f1:.2f}', fontsize=12, color='black')
plt.text(5.0, 2.2, f'Accuracy: {accuracy:.2f}', fontsize=12, color='black')

plt.title('График точек обучения и разделяющей линии', fontsize=16)
plt.xlabel('x1', fontsize=14)
plt.ylabel('x2', fontsize=14)
plt.legend()
plt.grid()
plt.savefig('task4_5_3.png')
plt.show()


