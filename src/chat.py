from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from search import search_prompt

load_dotenv()

INIT_QUESTION = "Introduce yourself and ask the user to make questions about the PDF document which you know about."

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

system = ("system", "You are a helpful assistant that answers the user questions based on a PDF using a local search engine.")
user = ("user", "{question}")

prompt = ChatPromptTemplate([system, user])
init_template = prompt.format_messages(question=INIT_QUESTION)


def main():
    try:
        intro_response = model.invoke(init_template)
        print("Chatbot: " + intro_response.content)
    except Exception as e:
        print(f"Erro ao inicializar o chat: {e}")
        return

    for i in range(5):
        question = input("User: ")
        if question.lower() == "exit":
            break
        try:
            prompt_text = search_prompt(question)
            response_chain = ChatPromptTemplate.from_template(
                prompt_text) | model | StrOutputParser()
            response = response_chain.invoke({})
            print("Chatbot: " + response)
        except Exception as e:
            print(f"Erro: {e}")
    print("Chat encerrado.")


if __name__ == "__main__":
    main()
