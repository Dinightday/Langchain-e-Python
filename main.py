from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]

llm = OpenAI(api_key=api_key)

numero_dia = 7
numero_crianca = 2
atividade = "musica"

prompt = f"Crie um roteiro de viagem de {numero_dia} dias, para uma familia com {numero_crianca} crianças e no carro indique uma {atividade}."

response = llm.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {
            "role": "system",
            "content": "Você é um assistente de viagem e quero que você crie itinerarios. Sem nenhuma outra complementação. Com base no prompt"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)
print(response.choices[0].message.content)