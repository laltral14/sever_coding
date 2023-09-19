from statistics import median

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import pandas as pd


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


def get_item_recommendations(data, user_id, num_recommendations=5):
    user_item_matrix, le_user, le_item, user_similarity = prepare_data(data)

    user_idx = le_user.transform([user_id])[0]

    # Находим средний рейтинг пользователя
    user_ratings = user_item_matrix.iloc[user_idx]
    user_mean_rating = user_ratings.mean()

    # Считаем взвешенное сходство между пользователями и их рейтингами
    weighted_similarities = user_similarity[user_idx] * (user_ratings - user_mean_rating)

    # Получаем индексы товаров, которые пользователь еще не покупал
    unrated_items_idx = user_ratings[user_ratings == 0].index

    # Сортируем товары по весу сходства и выбираем наиболее релевантные
    recommendations = unrated_items_idx[np.argsort(weighted_similarities[unrated_items_idx])[-num_recommendations:]]

    # Переводим индексы товаров обратно в их исходные значения
    recommended_items = le_item.inverse_transform(recommendations)

    return recommended_items
