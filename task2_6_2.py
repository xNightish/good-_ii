from matplotlib import pyplot as plt
import numpy as np

class MylineReg:
    # инициализация модели
    def __init__(self, n_iter=100, lm=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0, random_state=0, batch_size=10, nt=1):
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
        self.nt = nt
    
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
        Qe = self._calculate_metric(y_train, np.dot(x_train, self.weights))
        
        for i in range(self.n_iter):
            x_batch, y_batch = self.create_batch(x_train, y_train)
            Qk = self._calculate_metric(y_batch, np.dot(x_batch, self.weights))
            Qe = self.lm * Qk + (1 - self.lm) * Qe
            self.weights -= self._update_weights(x_batch, y_batch)

            self.loss.append(Qk)
            self.impurity.append(Qe)
            if verbose and (i + 1) % verbose == 0:
                print(f"Iteration {i + 1} | Loss: {Qk:.5f} | {self.metric}: {Qe:.5f}")
                
        Q = np.mean(self._calculate_metric(y_train, np.dot(x_train, self.weights)))
        return Qe, Q, self.weights
    
    # создание мини-батча
    def create_batch(self, x_train, y_train):
        start_index_bath = np.random.randint(0, x_train.shape[0]-self.batch_size-1)
        x_batch = x_train[start_index_bath:start_index_bath+self.batch_size]
        y_batch = y_train[start_index_bath:start_index_bath+self.batch_size]
        return x_batch, y_batch

    # производная логарифмической функции потерь по вектору w
    def _calculate_gradient(self, x, y):
        y_pred = np.dot(x, self.weights)
        grad = {
            'mae': lambda: np.mean(np.sign(y_pred - y)),
            'mse': lambda: 2 / x.shape[0] * np.dot(x.T, np.dot(x, self.weights) - y),
            'rmse': lambda: -(np.mean(y - y_pred)) / self._calculate_metric(y, y_pred),
            'mape': lambda: -100 * np.sign(y_pred - y).mean() / np.clip(np.mean(y), 1e-10, None),
            'r2': lambda: -2 * np.mean((y - y_pred) / np.sum((y - np.mean(y)) ** 2)),
            'logloss': lambda: np.mean([self._calculate_log_grad(x, y) for x, y in zip(x, y)], axis=0)
        }
        
        return grad.get(self.metric, lambda: None)()
    
    
    def _calculate_log_grad(self, x, y):
        M = np.dot(self.weights, x) * y
        return -(np.exp(-M) * x.T * y) / ((1 + np.exp(-M)) * np.log(2))
    
        
    # обновление весов
    def _update_weights(self, x, y):
        return self.nt * (self._calculate_gradient(x, y)
                        + self._calculate_regularization())
        
    # подсчет метрики
    def _calculate_metric(self, y_true, y_pred):
        metrics = {
            'mae': lambda: np.mean(np.abs(y_true - y_pred)),
            'mse': lambda: np.mean((y_true - y_pred) ** 2),
            'rmse': lambda: np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'mape': lambda: np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-10, None))) * 100,
            'r2': lambda: 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)),
            'logloss': lambda: np.mean(np.log2(1 + np.exp(-(y_true * y_pred))))
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
        

def func(x):
    return -0.5 * x ** 2 + 0.1 * x ** 3 + np.cos(3 * x) + 7


# модель
def model(w, x):
    xv = np.array([x ** n for n in range(len(w))])
    return w.T @ xv


coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

N = 5 # сложность модели (полином степени N-1)
lm_l1 = 2.0 # коэффициент лямбда для L1-регуляризатора
sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.zeros(N) # начальные нулевые значения параметров модели
n_iter = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)

x_train = np.array([[coord_x[i] ** n for n in range(N)] for i in range(sz)])
y_train = np.array(coord_y)


lm = MylineReg(n_iter=n_iter, lm=lm, random_state=0, batch_size=batch_size, metric='mse',reg="l1", l1_coef=lm_l1, l2_coef=2, nt=eta)
Qe, Q, w = lm.fit(x_train, y_train, verbose=0)


lm.plot_loss()
lm.plot_impurity()
lm.plot_decision_boundary()
print(w)
print(Qe)
print(Q)
