# Координаты первой точки
x1 = -3
y1 = 5

# Координаты второй точки
x2 = 7
y2 = 5

# Вычисление весов
w1 = (y2 - y1) / (x2 - x1)
w0 = w1 * (-x1) + y1
w = [w0, w1, -1]

# Вывод результата
print(w)