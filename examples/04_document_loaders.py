"""
In order to use document loaders:

pip install -qU langchain-community pypdf beautifulsoup4 lxml

pypdf is for PyPDFLoader
bs4 is for WebBaseLoader
lxml if you want a better html parser
"""
import os
import re
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, UnstructuredPDFLoader

pdf_file_path = os.path.join(os.getcwd(), "data", "machine_minds.pdf")

# Using the PyPDFLoader
pdf_loader = PyPDFLoader(pdf_file_path)

pages = []

# Each document is a Document object with page_content and metadata
documents = pdf_loader.load()

# Inspect the extracted documents
for i in range(2):
     print(f"Document: {i+1}\n")
     print(f"Metadata: {documents[i].metadata}\n")
     print(f"Content: {documents[i].page_content[:100]}...\n\n")



# Using WebBaseLoader
web_loader = WebBaseLoader("https://www.uu.se/en/centre/crb/news/archive/2024-09-23-exploring-artificial-consciousness-drawing-inspiration-from-the-human-brain")
web_loader.default_parser = "lxml"
web_loader.requests_kwargs = {"headers": {"User-Agent": "Mozilla/5.0"}} 
data = web_loader.load()

def sanitize_web_text(raw_text: str) -> str:
    """
    Remove repeated newlines
    Remove leading/trailing whitespace from each line
    """
    raw_text = re.sub(r'\n{2,}', '\n\n', raw_text)
    lines =  [line.strip() for line in raw_text.splitlines()]
    text = "\n".join(lines)
    return text

web_data = {
    "metadata": data[0].metadata,
    "content": sanitize_web_text(data[0].page_content)
}

print(f"SOURCE: {web_data['metadata']['source']}")
print(f"CONTENT: {web_data['content'][:100]}")


# Using UnstructuredPDFLoader to extract images and tables
adv_pdf_loader = UnstructuredPDFLoader(
	pdf_file_path,
	mode="elements",
	infer_table_structure=True,
)
pages = []

documents = adv_pdf_loader.load()

# Inspect the extracted documents
for doc in documents:
    print(f"Type: {doc.metadata['category']}, Content: {doc.page_content[:100]}...")
