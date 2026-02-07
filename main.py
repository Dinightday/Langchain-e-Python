from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]

class Destino(BaseModel):
    cidade: str = Field("A cidade a qual foi indicada.")
    motivo: str = Field("O motivo para essa cidade ser escolhida")

paseador = JsonOutputParser(pydantic_object=Destino)

llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.8)

numero_dia = 7
numero_crianca = 2
atividade = "musica"

prompt = PromptTemplate.from_template(
"""Crie um roteiro de viagem de {numero_di} dias, 
para uma familia com {numero_crianc} crianças e 
indique uma atividade: {atividad}.

{formato_saida}
""")

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"numero_di": numero_dia, 
                         "numero_crianc": numero_crianca, 
                         "atividad": atividade,
                         "formato_saida": paseador.get_format_instructions()})

print(response)