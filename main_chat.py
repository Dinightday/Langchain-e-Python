from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key,
    max_completion_tokens=256
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um guia de viagem especializado em destinos brasileiros"),
        ("placeholder", "{historico}"),
        ("human", "{query}")
    ]
)

chain = prompt | llm | StrOutputParser()

memoria = {}
sessao = "aula_langchain"

def historico_por_sessao(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

perguntas = [
    "Qual é a cor verdadeira do sol?",
    "Quem foi Pedro Alvares Cabral?"
]

cadeia_memoria = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico"
)

for pergunta in perguntas:
    response = cadeia_memoria.invoke(
        {
            "query": pergunta
        },
        config={"configurable": {"session_id": sessao}}
    )

    print(f"Usuário: {pergunta}")
    print(f"IA: {response}")
    print("-"*80)