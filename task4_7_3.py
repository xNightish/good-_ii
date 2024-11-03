# Начальные значения
TP = 87
FN = 13
TN = 150
FP = 50

precision = TP / (TP + FP)
recall = TP / (TP + FN)

print(f'precision = {precision}, recall = {recall}')