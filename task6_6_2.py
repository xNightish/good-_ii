import numpy as np
from sklearn.ensemble import RandomForestRegressor

x = np.arange(-3, 3, 0.1)
y = 0.3 * x + np.cos(2*x) + 0.2 * np.sin(7*x) # + np.random.normal(0.0, 0.1, n_samples)
x = x.reshape(-1, 1)

T = 5  # число деревьев

rf = RandomForestRegressor(max_depth=8, n_estimators=T, random_state=1)
rf.fit(x, y)

pr_y = rf.predict(x)

Q = ((pr_y - y) ** 2).mean()

print('Качество модели:', Q)

# Построение графика
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='red', label='Истинная функция f(x)', linewidth=2)
plt.plot(x, pr_y, color='blue', label='Аппроксимация', linewidth=2, linestyle='--')
plt.title('Аппроксимация функции f(x) случайным лесом')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
# plt.savefig('task6_6_2.png')
plt.show()