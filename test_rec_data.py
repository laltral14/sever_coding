import pandas as pd
import random

# Создаем случайные данные для 100 строк
rec_data_raw = {
    'id транзакции': list(range(1, 101)),
    'дата покупки': ['2023-09-01', '2023-09-02', '2023-09-03', '2023-09-04', '2023-09-05'] * 20,
    'список покупок': [random.sample(['Сахар', 'Стаканчики', 'Одноразовые палочки', 'Салфетки'], random.randint(1, 4)) for _ in range(100)],
    'список количества покупок': [random.sample(range(1, 11), len(items)) for items in data['список покупок']]
}

# Создаем датафрейм
rec_data = pd.DataFrame(rec_data_raw)