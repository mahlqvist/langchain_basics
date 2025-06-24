import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime as dt
from langchain.schema import Document
from collections import defaultdict


pdf_file_path = os.path.join(os.getcwd(), "data", "cognitive_architectures.pdf")

def adv_pdf_loader(file_path, title, authors, published):
	if published:
		date_format = '%Y-%m-%d'
		date_obj = dt.strptime(published, date_format)
		published_date = date_obj.date().isoformat()

	pdf_loader = UnstructuredPDFLoader(
		file_path=file_path,
		mode="elements",
		infer_table_structure=True,
	  )

	elements = pdf_loader.load()

	pages = defaultdict(list)
	for el in elements:
		page = el.metadata.get("page_number", 0)
		pages[page].append(el.page_content.strip())

	documents = []
	for page_num, texts in pages.items():
		full_text = "\n\n".join(texts)
		doc = Document(
			page_content=full_text,
			metadata={"source": "CoALA_Paper.pdf", "page": page_num}
		)
		documents.append(doc)


	for doc in documents:
		doc.metadata.update({'title': title, 'authors': authors, 'published': published_date})
		doc.metadata = {k: v for k, v in doc.metadata.items() if k in ["source", "page", "title", "authors", "published"]}
		
	return documents

raw_pdf_docs = adv_pdf_loader(pdf_file_path, "Cognitive Architectures for Language Agents", "Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, Thomas L. Griffiths", "2024-04-21")

# Create a CharacterTextSplitter with specific configuration:
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

pdf_chunks = text_splitter.split_documents(raw_pdf_docs)

# Define a function to display document statistics
def display_document_stats(docs, name):
	"""Display statistics about a list of document chunks"""
	total_chunks = len(docs)
	total_chars = sum(len(doc.page_content) for doc in docs)
	avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
	
	# Count unique metadata keys across all documents
	all_metadata_keys = set()
	for doc in docs:
		all_metadata_keys.update(doc.metadata.keys())
	
	# Print the statistics
	print(f"\n=== {name} Statistics ===")
	print(f"Total number of chunks: {total_chunks}")
	print(f"Average chunk size: {avg_chunk_size:.2f} characters")
	print(f"Metadata keys preserved: {', '.join(all_metadata_keys)}")
	
	if docs:
		for i in range(2):
			print(f"\nDocument: {i+1}\n")
			print(f"Metadata: {docs[i].metadata}\n")
			print(f"Content: {docs[i].page_content}\n\n")

# Display stats for chunks
display_document_stats(pdf_chunks, "Chunks Split With Improved PDF Loader")
