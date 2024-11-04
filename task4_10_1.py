import numpy as np

# Начальные значения для расчетов
TP = [45, 37, 51, 47]
TN = [36, 37, 29, 28]
FP = [18, 21, 15, 17]
FN = [8, 11, 9, 5]

# Рассчет precision и recall, используя микро-усреднение
precision = np.mean(TP) / (np.mean(TP) + np.mean(FP))
recall = np.mean(TP) / (np.mean(TP) + np.mean(FN))

print(f'precision = {precision}, recall = {recall}')
