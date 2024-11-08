import numpy as np

np.random.seed(0)
X = np.random.randint(0, 2, size=200)
S = lambda x: 1 - np.sum((np.bincount(x) / len(x)) ** 2)
t = 150

# Вычисление IG
S0, S_left, S_right = S(X), S(X[:t]), S(X[t:])
IG = S0 - (len(X[:t]) / len(X) * S_left + len(X[t:]) / len(X) * S_right)

# Вывод результатов
print("IG:", IG)

