import os
from dotenv import load_dotenv

from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

load_dotenv()

for k in [
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "DATABASE_URL",
        "PG_VECTOR_COLLECTION_NAME"
]:

    if not os.getenv(k):
        raise ValueError(f"{k} is not set in the environment variables")

PG_VECTOR_COLLECTION = os.getenv("PG_VECTOR_COLLECTION_NAME")
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


def __similarity_search(query: str, top_k: int = 5) -> list[str]:
    try:
        pg_vector = PGVector(
            collection_name=PG_VECTOR_COLLECTION,
            connection=DATABASE_URL,
            embeddings=EMBEDDINGS,
            use_jsonb=True
        )
        results = pg_vector.similarity_search(query, top_k=10)
        return [result.page_content for result in results]
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []


def search_prompt(question=None):
    contexto = "\n\n".join(__similarity_search(question))
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)
    return prompt
