import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Данные
X = [(166, 88), (147, 119), (133, 147), (113, 175), (91, 92), (120, 126), (146, 151), 
     (172, 174), (94, 192), (187, 193), (328, 82), (299, 97), (277, 131), (280, 171), 
     (299, 198), (348, 194), (378, 153), (372, 95), (222, 169), (332, 141), (69, 256), 
     (110, 258), (139, 257), (179, 256), (210, 256), (248, 256), (295, 256), (322, 254), 
     (350, 252), (377, 251), (400, 247), (403, 260), (378, 278), (341, 273), (306, 274), 
     (277, 275), (245, 274), (222, 275), (193, 276), (170, 276), (147, 279), (120, 274), 
     (91, 275), (65, 279)]

X = np.array(X)

# Кластеризация
clustering = DBSCAN(eps=55, min_samples=3, metric='euclidean')
res = clustering.fit_predict(X)

# Разделяем на выбросы и кластеры
Noise, X1, X2, X3 = [X[res == i] for i in range(-1, 3)]

# Визуализация
plt.figure(figsize=(10, 7))

# Отображение кластеров
clusters = [X1, X2, X3]
for i, cluster in enumerate(clusters):
    plt.scatter(cluster[:, 0], cluster[:, 1], s=100, label=f'Кластер {i + 1}')
    
# Отображение выбросов
plt.scatter(Noise[:, 0], Noise[:, 1], s=100, color='red', label='Выбросы')

plt.title('Кластеризация DBSCAN')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid()
plt.show()
