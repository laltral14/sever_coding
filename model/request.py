from pydantic import BaseModel


class Request(BaseModel):
    user_id: int
    login: str
    is_entrepreneur: bool


class SearchRequest(Request):
    text: str


class Recommendation(SearchRequest):
    good_id: int
