import numpy as np


def impurity(y):
    return sum((y - y.mean()) ** 2), len(y)


def inf_gain(y_left, y_right):
    HR1, len1 = impurity(y_left) if y_left.size else (0, 0)
    HR2, len2 = impurity(y_right) if y_right.size else (0, 0)
    return HR0 - ((len1 / len0) * HR1) - ((len2 / len0) * HR2)


x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4*x) + np.cos(2*x)
HR0, len0 = impurity(y)

th, IG = max(((t, inf_gain(y[x < t], y[x >= t])) 
              for t in x), key=lambda x: x[1])

print('Наилучший порог:', th)
print('IG:', IG)
