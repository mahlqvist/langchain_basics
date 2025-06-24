from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

file_name = "company_policies.txt"



def text_loader(file_name: str, title: str):

	file_path = os.path.join(os.getcwd(), "data", file_name)
	loader = TextLoader(file_path)
	documents = loader.load()
	
	for doc in documents:
		doc.metadata.update({'source': file_name, 'title': title})
		doc.metadata = {k: v for k, v in doc.metadata.items() if k in ["source", "title"]}

	return documents

docs = text_loader(file_name, "Company Policies")

text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=800,
	chunk_overlap=100,
	separators=[
		r"(?<=[.?!])\s+(?=[A-Z])",
		"\n\n",
		"\n",
		" ",
		""
	],
	is_separator_regex=True,
	keep_separator=True,
	strip_whitespace=True
)

text_chunks = text_splitter.split_documents(docs)

for i, chunk in enumerate(text_chunks):
	print(f"CHUNK: {i+1}\n")
	print(f"METADATA: {chunk.metadata}\n")
	print(f"CONTENT: {chunk.page_content}\n\n")