from statistics import median

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def get_prices(search, qdrant):
    search_res = qdrant.similarity_search_with_score(search, k=100, score_threshold=0.62, offset=1)
    prices = [item[0].metadata['цена покупки'] for item in search_res]

    # Вычисляем среднюю цену
    average_price = round(sum(prices) / len(prices))

    # Вычисляем медиану цен
    median_price = round(median(prices))

    return average_price, median_price


def prepare_data(data):
    # Преобразование категорий и товаров в числовые значения
    le_user = LabelEncoder()
    le_item = LabelEncoder()

    data['user_id'] = le_user.fit_transform(data['id покупателя'])
    data['item_id'] = le_item.fit_transform(data['id товара'])

    # Создание матрицы взаимодействий пользователь-товар
    user_item_matrix = pd.pivot_table(data, values='количество', index='user_id', columns='item_id', fill_value=0)

    # Вычисление сходства между пользователями на основе матрицы взаимодействий
    user_similarity = cosine_similarity(user_item_matrix)

    return user_item_matrix, le_user, le_item, user_similarity


def get_item_recommendations(rec_data, item_id):
    
    # Преобразование данных в бинарный формат (транзакции в столбцы с категориями)
    df_encoded = pd.get_dummies(rec_data['список покупок'].apply(pd.Series).stack()).sum(level=0)
    df_encoded = df_encoded.astype(bool).astype(int)
    
    # Задайте ID товара, для которого нужно получить рекомендации
    target_item = item_id
    
    # Подготовьте данные только для транзакций, в которых есть целевой товар
    df_target = df_encoded[df_encoded[target_item] == 1]
    
    # Применение алгоритма Apriori для поиска частых наборов
    frequent_itemsets = apriori(df_target, min_support=0.1, use_colnames=True)
    
    # Поиск ассоциативных правил
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
    
    # Фильтрация правил для исключения target_item из консеквентов
    filtered_rules = rules[~rules['consequents'].apply(lambda x: target_item in x)]
    
    # Отсортируйте правила по убыванию по показателю "lift"
    sorted_rules = filtered_rules.sort_values(by='lift', ascending=False)
    
    # Выведите рекомендации для заданного товара без target_item
    recommended_items = list(sorted_rules['consequents'].iloc[0])
    
    return recommended_items
