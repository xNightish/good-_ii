import numpy as np

# Начальные значения для расчетов
TP = [45, 37, 51, 47]
TN = [36, 37, 29, 28]
FP = [18, 21, 15, 17]
FN = [8, 11, 9, 5]

# Рассчет precision и recall, используя микро-усреднение
precision = np.mean([tp / (tp + fp) for tp, fp in zip(TP, FP)])
recall = np.mean([tp / (tp + fn) for tp, fn in zip(TP, FN)])

print(f'precision = {precision}, recall = {recall}')
