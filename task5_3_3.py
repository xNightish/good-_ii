import numpy as np

def nadaraya_watson_predict(rub_usd, h, num_days):
    current_day_index = len(rub_usd)

    for _ in range(num_days):
        # Вычисляем расстояния
        r = abs(current_day_index - np.arange(len(rub_usd))) / h
        
        # Вычисляем веса с использованием гауссовского ядра
        w = np.exp(-0.5 * r * r) / np.sqrt(2. * np.pi)
        
        # Прогнозируем значение для следующего дня
        y_pred = np.dot(rub_usd, w) / np.sum(w)
        
        # Обновляем массив rub_usd, добавляя предсказанное значение
        rub_usd = np.hstack((rub_usd, y_pred))
        current_day_index += 1  # Переходим к следующему дню
        
        yield y_pred  # Возвращаем предсказанное значение

# Начальные данные
rub_usd = np.array([75, 76, 79, 82, 85, 81, 83, 86, 87, 85, 83, 80, 77, 79, 78, 81, 84])
h = 3
num_days = 10  # Задаем количество дней для прогноза

# Создаем итератор для прогнозов
predictions_iterator = nadaraya_watson_predict(rub_usd, h, num_days)

# Получаем и выводим прогнозы
predict = list(predictions_iterator)
print(f"Прогноз курса на следующие {num_days} дней: {predict}")








        


