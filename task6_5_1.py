import numpy as np


def impurity(y):
    return sum((y - y.mean()) ** 2), len(y)


x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4*x) + np.cos(2*x)
t = 0 # порог

y_left = y[x < t]
y_right = y[x >= t]


HR1, len1 = impurity(y_left)
HR2, len2 = impurity(y_right)
HR0, len0 = impurity(y)

IG = HR0 - ((len1 / len0) * HR1) - ((len2 / len0) * HR2)

print('IG =', IG)
