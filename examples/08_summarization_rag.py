import requests
import codecs
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import uuid
import chromadb
import os


logging.basicConfig(level=logging.ERROR, filename="test.log", filemode="a", format="%(asctime)s from %(name)s: %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

env_path = os.path.join(os.getcwd(), "config", ".env")

_ = load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
	model="gpt-4.1-mini",
	temperature=0.3,
	top_p=0.9,
)

embeddings = OpenAIEmbeddings(
	model="text-embedding-3-small",
)


url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

url2 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/XVnuuEg94sAE4S_xAsGxBA.txt"

file_name = "company_policies.txt"

file_path = os.path.join(os.getcwd(), "data", file_name)

def download_file(url, file_path=None, return_data=None):
	headers = {
		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
		'Accept': '*/*',  # Accepts ANY file type
		'Connection': 'keep-alive',
		'Accept-Encoding': 'identity' 
	}
	try:
		response = requests.get(url, headers=headers, stream=True, timeout=30)
		response.raise_for_status()

		if return_data and not file_path:
			# Buffer to store downloaded bytes
			chunks = []

			# Download content in chunks
			for chunk in response.iter_content(chunk_size=8192):
				if chunk:  # filter out keep-alive chunks
					chunks.append(chunk)

			# Combine all chunks into one bytes object
			data_bytes = b''.join(chunks)

			data = codecs.decode(data_bytes, 'utf-8')
			return data

		with open(file_path, 'wb') as fh:
			for chunk in response.iter_content(chunk_size=8192):
				if chunk:
					fh.write(chunk)

		print("File downloaded successfully!")
	except requests.exceptions.RequestException as e:
		logging.error(f"Error downloading file: {e}")


if not os.path.isfile(file_path):
	download_file(url, file_path)

loader = TextLoader(file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=800,  
	chunk_overlap=0,
	length_function=len,
	separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""], 
	is_separator_regex=False
)

text_chunks = text_splitter.split_documents(documents)

vector_db = chromadb.PersistentClient(path="./chroma_test_storage")

policy_collection = vector_db.get_or_create_collection(name="company_policies")

documents = [chunk.page_content for chunk in text_chunks]
document_vectors = embeddings.embed_documents(documents)

policy_collection.upsert(
	documents=documents, 
	embeddings=document_vectors,
	metadatas=[chunk.metadata for chunk in text_chunks],  
	ids=[str(uuid.uuid4()) for _ in text_chunks]
)

def chroma_retriever(query: str, k: int=5):
	query_vector = embeddings.embed_query(query)
	
	results = policy_collection.query(
		query_embeddings=[query_vector],
		n_results=k,
		include=["documents", "metadatas"]
	)
	
	return results

template = """Use the information from the document to answer the question at the end. If you don't have enough information to answer the question do not try to make up an answer.
 
Context: {context}
Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": chroma_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Usage
response = rag_chain.invoke("Can I eat in company vehicles?")
print(response)