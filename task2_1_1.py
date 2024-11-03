import time
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 0.5 * x + 0.2 * x * x - 0.1 * x * x * x

def df(x):
    return 0.5 + 0.4 * x - 0.3 * x * x

N = 200 # число итераций
xx= -4 # начальное значение
lmd = 0.01 # шаг сходимости

x_plt = np.arange(-5, 5, 0.1)
f_plt = np.array([f(x) for x in x_plt])

plt.ion() # интерактивный режим
fig, ax = plt.subplots() # создание окна и осей графика
ax.grid(True) # включение сетки на графике

ax.plot(x_plt, f_plt, color='blue') # рисование графика
point = ax.scatter(xx, f(xx), color='red') # рисование точки на графике


for i in range(N):
    xx = xx - lmd * df(xx)
    point.set_offsets([xx, f(xx)]) # обновление точки
    fig.canvas.draw() # обновление графика
    fig.canvas.flush_events()
    time.sleep(0.02)
    
plt.ioff()
print(xx)
ax.scatter(xx, f(xx), color='blue')
plt.show()