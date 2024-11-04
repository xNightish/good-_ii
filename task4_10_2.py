import numpy as np

# Начальные значения для расчетов
TP = [57, 48, 60, 55]
TN = [32, 35, 28, 41]
FP = [13, 15, 12, 11]
FN = [7, 12, 11, 8]

# Рассчет precision и recall, используя макро-усреднение
precision = np.mean([tp / (tp + fp) for tp, fp in zip(TP, FP)])
recall = np.mean([tp / (tp + fn) for tp, fn in zip(TP, FN)])

print(f'precision = {precision}, recall = {recall}')