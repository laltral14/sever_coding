from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config.cfg import OPENAI_API_KEY

prompt_template = """Ты помошник в выборе товаров. На вход подается необходимый товар.
Вежливо предложи (без вопросов) покупателю подходящий товар.
Проанализируй список словарей и подбери товар, который больше всего соответствует запросу.
Проверь по названию, точно ли это ищет покупатель.
В начале сообщения должно стоять вам подходит {название товара}. 
"""


def search(qdrant, search='Купить кошачий корм без глютена'):
    # Создание объекта PromptTemplate. Он будет использоваться для вставки переменных в промпт.
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=['название товара']
    )

    # Создание цепочки для взаимодействия с моделью.
    chain = LLMChain(
        llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY,
                   max_tokens=500),
        prompt=PROMPT
    )

    # Выполнение поиска товаров в Qdrant, основываясь на запросе.
    relevants = qdrant.similarity_search(search, k=7)

    # Извлечение метаданных (информации о товарах) из результатов поиска.
    doc = [relevant.dict()['metadata'] for relevant in relevants]

    # Запуск цепочки для обработки и анализа списка найденных товаров.
    return chain.run(doc)
