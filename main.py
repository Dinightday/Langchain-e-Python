from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.8)

numero_dia = 7
numero_crianca = 2
atividade = "musica"

prompt = f"Crie um roteiro de viagem de {numero_dia} dias, para uma familia com {numero_crianca} crianças e no carro indique uma {atividade}."

response = llm.invoke(prompt)

print(response.content)