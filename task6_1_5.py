import numpy as np

x_test = [(9, 6), (2, 4), (-3, -1), (3, -2), (-3, 6), (7, -3), (6, 2)]
x_test = np.array(x_test)

# точки на разделяющей прямой
x1, y1 = 7, 7
x2, y2 = 2, 0

# вычисление весов
w1 = (y2 - y1)/ (x2 - x1)
w0 = w1 * (-x1) + y1
w = np.array([w0, w1, -1])

# предсказание
X = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
predict = -np.sign(w @ X.T)

# вывод результата
print(predict)

