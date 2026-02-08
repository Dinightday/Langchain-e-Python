from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.globals import get_debug
from typing import Optional
from operator import itemgetter
import os

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]

class Destino(BaseModel):

# Cidade
    cidade: Optional[str] = Field(description="O nome da cidade em que foi recomendada")
    motivo: Optional[str] = Field(description="O motivo o porquê essa cidade foi escolhida para o usuario")
    score: Optional[float] = Field(description="Score do nivel de confiança de 0 a 1")

class Restaurante(BaseModel):

# Restaurantes
    restaurante: Optional[str] = Field(description="Algum restaurante muito bom para comer na cidade")
    motivo: Optional[str] = Field(description="O motivo pelo restaurante ser escolhio")
    score: Optional[float] = Field(description="Score do nivel de confiança de 0 a 1")

passeador_destino = JsonOutputParser(pydantic_object=Destino)
passeador_restaurante = JsonOutputParser(pydantic_object=Restaurante)

llm = ChatOpenAI(api_key=api_key, 
                model="gpt-3.5-turbo",
                temperature=0.2)

prompt1 = PromptTemplate.from_template(
"""Sugira uma cidade que cobre meu interesse: {interesse}
Formatação: {parser1}
""",
partial_variables={"parser1": passeador_destino.get_format_instructions()})

prompt2 = PromptTemplate.from_template(
"""Me de algumas recomendações de restaurante na cidade {cidade}.

Formatação: {parser2}
""",
partial_variables={"parser2": passeador_restaurante.get_format_instructions()})

prompt3 = PromptTemplate.from_template("""Sugira atividades culturais na cidade: {cidade}""")

chain1 = prompt1 | llm | passeador_destino
chain2 = prompt2 | llm | passeador_restaurante
chain3 = prompt3 | llm | StrOutputParser()

chain_tripla = (chain1 | {"cidade": itemgetter("cidade")} | chain2 | {"cidade": itemgetter("restaurante")} | chain3)

response = chain_tripla.invoke(
    {
        "interesse": "praia"
    }
)


print(response)