from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.globals import get_debug
from typing import Optional
import os

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]

class Destino(BaseModel):

# Cidade
    ruim_bom: Optional[str] = Field(description="Responda se a cidade é Boa ou Ruim para turismo. Apenas Boa ou Ruim")
    motivo: Optional[str] = Field(description="O motivo o porquê essa cidade foi escolhida para o usuario")
    score: Optional[float] = Field(description="Score do nivel de confiança de 0 a 1")

class Restaurante(BaseModel):

# Restaurantes
    restaurante: Optional[str] = Field(description="Algum restaurante muito bom para comer")
    motivo: Optional[str] = Field(description="O motivo pelo restaurante ser escolhio")
    score: Optional[float] = Field(description="Score do nivel de confiança de 0 a 1")

passeador_destino = JsonOutputParser(pydantic_object=Destino)
passeador_restaurante = JsonOutputParser(pydantic_object=Restaurante)

llm = ChatOpenAI(api_key=api_key, 
                model="gpt-3.5-turbo",
                temperature=0.2)

interesse = "Paraty (RJ)"

prompt1 = PromptTemplate.from_template(
""" Quero ir para {cidade}. Me de algumas recomendações.
    Se for ruim me traz uma cidade mais proxima que seja boa e do mesmo país.
Formatação: {parser}
""")

prompt2 = PromptTemplate.from_template(
"""Me de algumas recomendações de restaurante na cidade {cidade}.

Formatação: {parser}
""")

chain_destino = prompt1 | llm | passeador_destino

chain_restaurante = prompt2 | llm | passeador_restaurante

chain_dupla = (chain_destino | chain_restaurante)

response = chain_destino.invoke({
    "cidade": interesse,
    "parser": chain_dupla
})


print(response)