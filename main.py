from fastapi import FastAPI

from model.request import SearchRequest, Recommendation
from services.qdrant import QdrantSupplier
from services.recommendation import get_prices, get_item_recommendations
from services.search import search
from test_data import data

app = FastAPI()
# тестовый пример
qdrant_supplier = QdrantSupplier(data, 'описание товара', 'history')


@app.get("/search")
def search_(search_request: SearchRequest):
    return search(qdrant_supplier.qdrant, search_request.text)


@app.get("/prices")
def prices(recommendation_request: Recommendation):
    return get_prices(recommendation_request.text, qdrant_supplier.qdrant)


@app.get("/items")
def items(recommendation_request: Recommendation):
    return get_item_recommendations(data, recommendation_request.user_id)
