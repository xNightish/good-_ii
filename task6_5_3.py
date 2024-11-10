import numpy as np
from sklearn import tree

x = np.arange(-2, 3, 0.1).reshape(-1, 1)
y = 0.3 * x ** 2 - 0.2 * x ** 3 - 0.5 * np.sin(4*x)

clf = tree.DecisionTreeRegressor(max_depth=4)
clf.fit(x, y)

pr_y = clf.predict(x).reshape(-1, 1)
print(pr_y)
print(y)

Q= ((pr_y - y) ** 2).mean()

print('Качество модели:', Q)

# Построение графика
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='red', label='Истинная функция f(x)', linewidth=2, alpha=0.5)
plt.plot(x, pr_y, color='blue', label='Аппроксимация', linewidth=2, linestyle='--')
plt.title('Аппроксимация функции f(x) деревом решений')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
# plt.savefig('task6_5_3.png')
plt.show()
