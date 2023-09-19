from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant

from config.cfg import *


class QdrantSupplier:

    def __init__(self, data, content, vb_name, OA=True):
        # Загружаем фрейм данных в лоадер, указывая колонку для векторизации.
        loader = DataFrameLoader(data, page_content_column=content)
        documents = loader.load()

        # Создаем сплиттер документов для разделения текста на более мелкие части (фрагменты).
        # В данном случае, мы разбиваем текст на фрагменты размером 1000 токенов с некоторым перекрытием.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        if OA == False:
            # Если OA (OpenAI) равно False, то мы используем Hugging Face для векторизации.
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

            # Создаем экземпляр Qdrant, который будет использоваться для хранения и поиска векторов.
            qdrant = Qdrant.from_documents(
                texts,
                embeddings,
                url=QDRANT_HOST,
                prefer_grpc=True,
                api_key=QDRANT_API_KEY,
                collection_name=vb_name,
                force_recreate=True
            )
            qdrant.as_retriever()
        else:
            # Если OA (OpenAI) равно True, то мы используем OpenAI для векторизации.
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

            # Создаем экземпляр Qdrant аналогично предыдущему случаю.
            qdrant = Qdrant.from_documents(
                texts,
                embeddings,
                url=QDRANT_HOST,
                prefer_grpc=True,
                api_key=QDRANT_API_KEY,
                collection_name=vb_name,
                force_recreate=True
            )

        # Возвращаем экземпляр qdrant, который используется для поиска векторов.
        self.qdrant = qdrant
