import os
import logging
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# * Não consegui fazer o modelo abaixo funcionar gratuitamente
# * Erro: openai.RateLimitError: Error code: 429 - insufficient_quota
# from langchain_openai import OpenAIEmbeddings
# EMBEDDINGS = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

for k in [
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "DATABASE_URL",
        "PG_VECTOR_COLLECTION_NAME",
        "PDF_PATH"]:

    if not os.getenv(k):
        raise ValueError(f"{k} is not set in the environment variables")


PG_VECTOR_COLLECTION = os.getenv("PG_VECTOR_COLLECTION_NAME")
DATABASE_URL = os.getenv("DATABASE_URL")
PDF = os.getenv("PDF_PATH")
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


def __get_pdf_path(pdf: str) -> str:
    try:
        logging.info(f"Obtendo caminho do PDF: {pdf}")
        current_dir = Path(__file__).parent
        root_dir = current_dir.parent
        pdf_path = str(root_dir / pdf)
        logging.info(f"Caminho do PDF obtido: {pdf_path}")
        return pdf_path
    except Exception as e:
        logging.error(f"Erro ao obter caminho do PDF: {e}")
        raise


def __load_and_split(pdf_path: str) -> list[Document]:
    try:
        logging.info(f"Carregando e dividindo documento: {pdf_path}")
        doc = PyPDFLoader(pdf_path).load()

        if not doc:
            raise Exception("Failed to load document")

        splitted = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            add_start_index=False
        ).split_documents(doc)

        if not splitted:
            raise Exception("Failed to split document")

        logging.info(
            f"Documento carregado e dividido em {len(splitted)} chunks")
        return splitted
    except Exception as e:
        logging.error(f"Erro ao carregar e dividir documento: {e}")
        raise


def __enrich_documents(
        docs: list[Document]) -> tuple[list[Document], list[str]]:
    try:
        logging.info(f"Enriquecendo {len(docs)} documentos")
        enriched = [
            Document(
                page_content=doc.page_content,
                metadata={
                    k: v for k, v in doc.metadata.items() if v not in [
                        None, ""]
                },
            )
            for doc in docs
        ]

        ids = [f"doc-{i}" for i in range(len(enriched))]
        logging.info(f"Documentos enriquecidos com {len(ids)} IDs")
        return enriched, ids
    except Exception as e:
        logging.error(f"Erro ao enriquecer documentos: {e}")
        raise


def __store_docs(
    docs: list[Document],
    ids: list[str],
    embeddings: HuggingFaceEmbeddings,
    pg_vector_collection: str,
    database_url: str
) -> None:
    try:
        logging.info(
            f"Armazenando {len(docs)} documentos na coleção {pg_vector_collection}")
        store = PGVector(
            embeddings=embeddings,
            collection_name=pg_vector_collection,
            connection=database_url,
            use_jsonb=True
        )

        store.add_documents(documents=docs, ids=ids)
        logging.info("Documentos armazenados com sucesso")
    except Exception as e:
        logging.error(f"Erro ao armazenar documentos: {e}")
        raise


def ingest_pdf(ingest_pdf_params: dict[str, str]) -> None:
    try:
        logging.info("Iniciando ingestão do PDF")
        pdf_path = __get_pdf_path(ingest_pdf_params["pdf"])
        splitted = __load_and_split(pdf_path)
        enriched, ids = __enrich_documents(splitted)
        __store_docs(
            enriched,
            ids,
            ingest_pdf_params["embeddings"],
            ingest_pdf_params["pg_vector_collection"],
            ingest_pdf_params["database_url"])
        logging.info("Ingestão do PDF concluída com sucesso")
    except Exception as e:
        logging.error(f"Erro durante a ingestão do PDF: {e}")
        raise


if __name__ == "__main__":
    try:
        logging.info("Iniciando execução do script ingest.py")
        ingest_pdf_params = {
            "pdf": PDF,
            "embeddings": EMBEDDINGS,
            "pg_vector_collection": PG_VECTOR_COLLECTION,
            "database_url": DATABASE_URL
        }

        ingest_pdf(ingest_pdf_params)
        logging.info("Execução do script concluída com sucesso")
    except Exception as e:
        logging.error(f"Erro na execução do script: {e}")
        raise
