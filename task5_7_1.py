import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# Ваши данные
X = [(189, 185), (172, 205), (156, 221), (154, 245), (164, 265), (183, 275), (204, 276),
     (227, 271), (241, 255), (250, 229), (240, 197), (217, 183), (194, 202), (179, 224), (179, 248), (199, 249),
     (197, 227), (211, 214), (211, 242), (210, 265), (226, 237), (218, 196), (79, 106), (97, 132), (117, 159), 
     (138, 174), (148, 163), (140, 145), (121, 123), (112, 108), (89, 92), (282, 162), (298, 180), (344, 154),
     (344, 113), (362, 67), (397, 77), (412, 121), (379, 112), (377, 148), (312, 130)]

X = np.array(X)

# Кластеризация
K = 3
clustering = AgglomerativeClustering(n_clusters=K, linkage="ward", metric="euclidean")
res = clustering.fit_predict(X)

# Разделяем на кластеры
X1, X2, X3 = [X[res == i] for i in range(K)]

# Создание дендрограммы
linked = linkage(X, method='ward')
# Настройка подграфиков
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
# График с точками
for i in range(K):
    ax1.scatter(X[res == i][:, 0], X[res == i][:, 1], label=f'Класс {i + 1}')
# Подписи точек
for i, txt in enumerate(range(len(X))):
    ax1.annotate(txt, (X[i, 0], X[i, 1]), fontsize=8, ha='right')
ax1.set_title("Кластеризация")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()
# Дендограмма
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, ax=ax2, color_threshold=300) # 300 - расстояние между кластерами
ax2.axhline(y=300, color='r', linestyle='--') # линия разделяющая кластеры
ax2.set_xlabel("Индексы образцов")
ax2.set_ylabel("Расстояние")
ax2.set_title("Дендограмма")
plt.tight_layout()
# plt.savefig('task5_7_1.png')
plt.show()
