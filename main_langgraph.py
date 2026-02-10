from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.caches import InMemoryCache
from dotenv import load_dotenv
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio
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

roteador = prompt_roteador | client.with_structured_output(Rota)

class Estado(TypedDict):
    query: str
    destino: str
    resposta: str

async def no_roteador(estado: Estado, config=RunnableConfig):
    return {"destino": await roteador.ainvoke({"query": estado["query"]}, config)}
async def no_praia(estado: Estado, config=RunnableConfig):
    return {"resposta": await chain_praia.ainvoke({"query": estado["query"]}, config)}
async def no_montanha(estado: Estado, config=RunnableConfig):
    return {"resposta": await chain_montanha.ainvoke({"query": estado["query"]}, config)}

def escolher_no(estado:Estado) -> Literal["praia", "montanha"]:
    if estado["destino"]["destino"] == "praia":
        return "praia"
    else:
        return "montanha"

grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke(
        {
            "query": "Quero visitar um lugar no Brasil famoso por praias e culturas"
        }
    )
    print(resposta["resposta"])

asyncio.run(main())