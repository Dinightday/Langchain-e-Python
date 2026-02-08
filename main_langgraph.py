from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.caches import InMemoryCache
from dotenv import load_dotenv
from typing import Literal, TypedDict
import os

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]

client = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

prompt_praia = ChatPromptTemplate(
    [
        ("system", "Você é um consultor de viagens para a praia."),
        ("human", "{query}")
    ]
)

prompt_montanha = ChatPromptTemplate(
    [
        ("system", "Você é um consultor de viagens para montanhas."),
        ("human", "{query}")
    ]
)

chain_praia = prompt_praia | client | StrOutputParser()
chain_montanha = prompt_montanha | client | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate(
    [
        ("system", "Escolha com base na vontade do usuario"),
        ("human", "{query}")
    ]
) 

chain_roteadora = prompt_roteador | client.with_structured_output(Rota)

def response(pergunta : str):
    rota = chain_roteadora.invoke(
        {
            "query": pergunta
        }
    )

    if rota["destino"] == "praia":
        return chain_praia.invoke({"query": pergunta})
    elif rota["destino"] == "montanha":
        return chain_montanha.invoke({"query": pergunta})

print(response("Quero me divertir em um lugar quente no Brasil."))