from matplotlib import pyplot as plt
import numpy as np

class MylineReg:
    # инициализация модели
    def __init__(self, n_iter=100, lm=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0, random_state=0, batch_size=10):
        self.n_iter = n_iter
        self.lm = lm
        self.weights = weights
        self.metric = metric
        self.best_score = None 
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.random_state = random_state
        self.batch_size = batch_size
        self.loss = []
        self.impurity = []
    
    # возвращает строковое представление объекта
    def __str__(self):
        return (
            f"MyLineReg(n_iter={self.n_iter}, learning_rate={self.lm}, "
            f"metric={self.metric}, reg={self.reg}, l1_coef={self.l1_coef}, "
            f"l2_coef={self.l2_coef}, random_state={self.random_state})"
        )
    
    # обучение модели
    def fit(self, x_train, y_train, verbose=0):
        self.x_train = x_train
        self.y_train = y_train
        np.random.seed(self.random_state)
        self.weights = np.zeros(x_train.shape[1])
        self.nt = np.array([0.5] + [0.01] * (len(self.weights) - 1))
        Qe = np.mean(self._calculate_log_loss(x_train, y_train))
        
        for i in range(self.n_iter):
            x_batch, y_batch = self.create_batch(x_train, y_train)
            Qk = self._calculate_log_loss(x_batch, y_batch).mean()
            Qe = self.lm * Qk + (1 - self.lm) * Qe
            self.weights -=  self._update_weights(x_batch, y_batch)
            metric = self._calculate_metric(y_batch, np.dot(x_batch, self.weights))
            self.loss.append(Qk)
            self.impurity.append(Qe)
            if verbose and (i + 1) % verbose == 0:
                print(f"Iteration {i + 1} | Loss: {Qk:.5f} | {self.metric}: {metric:.5f}")
                
        Q = (self.weights @ x_train.T * y_train < 0).mean()
        return Qe, Q, self.weights
    
    # создание мини-батча
    def create_batch(self, x_train, y_train):
        start_index_bath = np.random.randint(0, x_train.shape[0]-self.batch_size-1)
        x_batch = x_train[start_index_bath:start_index_bath+self.batch_size]
        y_batch = y_train[start_index_bath:start_index_bath+self.batch_size]
        return x_batch, y_batch

    # логарифмическая функция потерь
    def _calculate_log_loss(self, x, y):
        M = np.dot(x, self.weights) * y
        return np.log2(1 + np.exp(-M))

    # производная логарифмической функции потерь по вектору w
    def _calculate_log_gradient(self, x, y):
        M = np.dot(self.weights, x) * y
        return -(np.exp(-M) * x.T * y) / ((1 + np.exp(-M)) * np.log(2))
    
    # обновление весов
    def _update_weights(self, x, y):
        return self.nt * (np.mean([self._calculate_log_gradient(x, y) for x, y in zip(x, y)], axis=0)
                        + self._calculate_regularization())
        
    # подсчет метрики
    def _calculate_metric(self, y_true, y_pred):
        metrics = {
            'mae': lambda: np.mean(np.abs(y_true - y_pred)),
            'mse': lambda: np.mean((y_true - y_pred) ** 2),
            'rmse': lambda: np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'mape': lambda: np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,  # Добавлен epsilon для предотвращения деления на ноль
            'r2': lambda: 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        }
        return metrics.get(self.metric, lambda: None)()
    
    # получение коэффициентов весов
    def get_coef(self):
        return self.weights
    
    # регуляризация
    def _calculate_regularization(self):
        gradient_reg = {
            'l1': lambda: self.l1_coef * np.sign(np.array([0, *self.weights[1:]])),
            'l2': lambda: 2 * self.l2_coef * np.array([0, *self.weights[1:]]),
            'elasticnet': lambda: (self.l1_coef * np.sign(np.array([0, *self.weights[1:]])) +
                                   2 * self.l2_coef * np.np.array([0, *self.weights[1:]])),
        }
            
        return gradient_reg.get(self.reg, lambda: 0)()
    
    # предсказание
    def predict(self, x_test):
        return np.dot(x_test, self.weights)
    
    
    # график функции потерь
    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss, color='blue', linewidth=2, label='Функция потерь')
        plt.title('График функции потерь', fontsize=16)
        plt.xlabel('Итерация', fontsize=14)
        plt.ylabel('Потеря', fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()

    # график среднего имперического риска
    def plot_impurity(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.impurity, color='orange', linewidth=2, label='Имперический риск')
        plt.title('График среднего имперического риска', fontsize=16)
        plt.xlabel('Итерация', fontsize=14)
        plt.ylabel('Имперический риск', fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()

    # график точек обучения и разделяющей линии
    def plot_decision_boundary(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(x_train[:, 1], x_train[:, 2], c=y_train, cmap='bwr', edgecolor='k', label='Точки обучения')
        x = np.linspace(2.8, 7, 100)
        y = -self.weights[0] / self.weights[2] - self.weights[1] / self.weights[2] * x
        plt.plot(x, y, color='green', linewidth=2, label='Разделяющая линия')
        plt.title('График точек обучения и разделяющей линии', fontsize=16)
        plt.xlabel("x1", fontsize=14)
        plt.ylabel("x2", fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()
        

data_x = [(5.8, 1.2), (5.6, 1.5), (6.5, 1.5), (6.1, 1.3), (6.4, 1.3), (7.7, 2.0), (6.0, 1.8), (5.6, 1.3), (6.0, 1.6), (5.8, 1.9), (5.7, 2.0), (6.3, 1.5), (6.2, 1.8), (7.7, 2.3), (5.8, 1.2), (6.3, 1.8), (6.0, 1.0), (6.2, 1.3), (5.7, 1.3), (6.3, 1.9), (6.7, 2.5), (5.5, 1.2), (4.9, 1.0), (6.1, 1.4), (6.0, 1.6), (7.2, 2.5), (7.3, 1.8), (6.6, 1.4), (5.6, 2.0), (5.5, 1.0), (6.4, 2.2), (5.6, 1.3), (6.6, 1.3), (6.9, 2.1), (6.8, 2.1), (5.7, 1.3), (7.0, 1.4), (6.1, 1.4), (6.1, 1.8), (6.7, 1.7), (6.0, 1.5), (6.5, 1.8), (6.4, 1.5), (6.9, 1.5), (5.6, 1.3), (6.7, 1.4), (5.8, 1.9), (6.3, 1.3), (6.7, 2.1), (6.2, 2.3), (6.3, 2.4), (6.7, 1.8), (6.4, 2.3), (6.2, 1.5), (6.1, 1.4), (7.1, 2.1), (5.7, 1.0), (6.8, 1.4), (6.8, 2.3), (5.1, 1.1), (4.9, 1.7), (5.9, 1.8), (7.4, 1.9), (6.5, 2.0), (6.7, 1.5), (6.5, 2.0), (5.8, 1.0), (6.4, 2.1), (7.6, 2.1), (5.8, 2.4), (7.7, 2.2), (6.3, 1.5), (5.0, 1.0), (6.3, 1.6), (7.7, 2.3), (6.4, 1.9), (6.5, 2.2), (5.7, 1.2), (6.9, 2.3), (5.7, 1.3), (6.1, 1.2), (5.4, 1.5), (5.2, 1.4), (6.7, 2.3), (7.9, 2.0), (5.6, 1.1), (7.2, 1.8), (5.5, 1.3), (7.2, 1.6), (6.3, 2.5), (6.3, 1.8), (6.7, 2.4), (5.0, 1.0), (6.4, 1.8), (6.9, 2.3), (5.5, 1.3), (5.5, 1.1), (5.9, 1.5), (6.0, 1.5), (5.9, 1.8)]
data_y = [-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1]

x_train = np.array([[1, x[0], x[1], 0.8*x[0], (x[0]+x[1])/2] for x in data_x])
y_train = np.array(data_y)


lm = MylineReg(n_iter=500, lm=0.01, random_state=0, batch_size=10, metric="mse",reg="l1", l1_coef=0.05, l2_coef=0.05)
Qe, Q, w = lm.fit(x_train, y_train, verbose=100)

# lm.plot_loss()
# lm.plot_impurity()
# lm.plot_decision_boundary()
print(w)
print(Qe)
print(Q)
