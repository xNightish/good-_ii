TP = 65
FN = 5
TN = 120
FP = 30

precision = TP / (TP + FP)
recall = TP / (TP + FN)

print(f'precision = {precision}, recall = {recall}')