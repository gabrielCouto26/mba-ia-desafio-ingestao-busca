import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

load_dotenv()

for k in [
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_MODEL",
        "DATABASE_URL",
        "PG_VECTOR_COLLECTION_NAME",
        "PDF_PATH"]:

    if not os.getenv(k):
        raise ValueError(f"{k} is not set in the environment variables")


EMBEDDINGS = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))
PG_VECTOR_COLLECTION = os.getenv("PG_VECTOR_COLLECTION_NAME")
DATABASE_URL = os.getenv("DATABASE_URL")
PDF = os.getenv("PDF_PATH")


def __get_pdf_path(pdf: str) -> str:
    current_dir = Path(__file__).parent
    return str(current_dir / pdf)


def __load_and_split(pdf_path: str) -> list[Document]:
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

    return splitted


def __enrich_documents(
        docs: list[Document]) -> tuple[list[Document], list[str]]:
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

    return enriched, ids


def __store_docs(
    docs: list[Document],
    ids: list[str],
    embeddings: OpenAIEmbeddings,
    pg_vector_collection: str,
    database_url: str
) -> None:
    store = PGVector(
        embeddings=embeddings,
        collection_name=pg_vector_collection,
        connection=database_url,
        use_jsonb=True
    )

    store.add_documents(documents=docs, ids=ids)


def ingest_pdf(ingest_pdf_params: dict[str, str]) -> None:
    pdf_path = __get_pdf_path(ingest_pdf_params["pdf"])
    splitted = __load_and_split(pdf_path)
    enriched, ids = __enrich_documents(splitted)
    __store_docs(
        enriched,
        ids,
        ingest_pdf_params["embeddings"],
        ingest_pdf_params["pg_vector_collection"],
        ingest_pdf_params["database_url"])


if __name__ == "__main__":
    ingest_pdf_params = {
        "pdf": PDF,
        "embeddings": EMBEDDINGS,
        "pg_vector_collection": PG_VECTOR_COLLECTION,
        "database_url": DATABASE_URL
    }

    ingest_pdf(ingest_pdf_params)
