import numpy as np
import matplotlib.pyplot as plt

# Координаты четырех точек
x = np.array([0, 1, 2, 3])
y = np.array([0.5, 0.8, 0.6, 0.2])

# Множество точек для промежуточного восстановления функции
x_est = np.arange(0, 3.1, 0.1)

# Метрика расстояния
p = lambda x, x_k: np.abs(x_k - x)

# Функции ядра
K_tr = lambda r: (1 - np.abs(r)) * (np.abs(r) <= 1)  # Треугольное окно
K_g = lambda r: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * r**2)  # Гауссово ядро
K_ep = lambda r: 3/4 * (1 - r**2) * (np.abs(r) <= 1)  # Ядро Епанечникова
K_q = lambda r: 15/16 * (1 - r**2) ** 2 * (np.abs(r) <= 1)  # Ядро квадратное
K_pr = lambda r: 1/2 * (np.abs(r) <= 1)  # Ядро прямоугольное

Ks = [K_tr, K_g, K_ep, K_q, K_pr]
kernel_names = ['Треугольное', 'Гауссово', 'Епанечникова', 'Квадратное', 'Прямоугольное']

# Создаем графики для каждой функции ядра
for K, name in zip(Ks, kernel_names):
    # Создаем график
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Значения h для разных подграфиков
    h_values = [0.1, 0.3, 1, 10]
    
    for ax, h in zip(axs.flatten(), h_values):
        # Векторизованное вычисление весов
        weights = K(p(x_est[:, np.newaxis], x) / h)  # Формируем массив весов

        # Вычисление суммы весов и взвешенной суммы
        weights_sum = np.sum(weights, axis=1)
        weighted_sum = np.nansum(weights * y, axis=1)

        # Вычисление y_est с обработкой деления на ноль
        with np.errstate(divide='ignore', invalid='ignore'):
            y_est = weighted_sum / weights_sum 

        # Заменяем NaN и inf на 0
        y_est[np.isnan(y_est)] = 0
        y_est[np.isinf(y_est)] = 0

        # Визуализация
        ax.plot(x_est, y_est, label=f'Восстановленная функция (h={h})', color='blue')
        ax.scatter(x, y, color='red', label='Исходные точки')
        ax.set_title(f'{name} ядро: Восстановление функции с h={h}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()



# Вычисление восстановленной функции
h = 1

# Векторизованное вычисление весов
weights = K(p(x_est[:, np.newaxis], x) / h)  # Формируем массив весов

# Рассчитываем восстановленные значения функции
y_est = np.sum(weights * y, axis=1) / np.sum(weights, axis=1)