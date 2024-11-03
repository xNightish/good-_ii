import numpy as np

x1, y1 = -1, 4
x2, y2 = 5, 0

w1 = (y2 - y1)/ (x2 - x1)
w0 = w1 * (-x1) + y1
w = np.array([w0, w1, -1])
print(w)
