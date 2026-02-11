from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

arquivos_pdf = [
    "documentos/GTB_platinum_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf",
    "documentos/GTB_standard_Nov23.pdf"
]

documentos = sum(
    [
        PyPDFLoader(arquivo).load() for arquivo in arquivos_pdf], 
        []
)

api_key = os.environ["OPENAI_API_KEY"]

cliente_openai = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0.3
)

embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


corte = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=100
).split_documents(documents=documentos)

search = FAISS.from_documents(
    corte, embed
).as_retriever(search_kwargs = {"k": 2})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda exclusivamente o conteúdo fornecido"),
        ("human", "{query}\n\nContexto: {procura}\n\nResposta:")
    ]
)

chain = prompt | cliente_openai | StrOutputParser()


def response(pergunta:str):
    guardar = []
    trecho = search.invoke(pergunta)
    for trechinho in trecho:
        guardar.append(trechinho)

    return chain.invoke(
        {
            "query": pergunta,
            "procura": guardar
        }
    )
print(response("Como devo proceder se eu tiver um item roubado?"))