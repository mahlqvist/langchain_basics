import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from datetime import datetime as dt
import re


pdf_file_path = os.path.join(os.getcwd(), "data", "cognitive_architectures.pdf")

def document_loader(file_path, title, authors, published):
    if published:
        date_format = '%Y-%m-%d'
        date_obj = dt.strptime(published, date_format)
        published_date = date_obj.date().isoformat()

    pdf_loader = PyPDFLoader(file_path)

    documents = pdf_loader.load()

    for doc in documents:
        doc.metadata.update({'title': title, 'authors': authors, 'published': published_date})
        doc.metadata = {k: v for k, v in doc.metadata.items() if k in ["page", "title", "authors", "published"]}
        
    return documents

raw_pdf_docs = document_loader(pdf_file_path, "Machine Minds: The Blueprint of Artificial Consciousness", "Sidharta Chatterjee", "2024-06-21")

def web_loader(web_url):
    web_loader = WebBaseLoader(
        web_path=web_url,
        default_parser="lxml",
        requests_kwargs = {"headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}}
    )

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

    # Split by natural sections (e.g., double newlines)
    text = sanitize_web_text(data[0].page_content)
    sections = [s.strip() for s in text.split("\n\n") if s.strip()]

    # Convert to Documents and apply recursive splitting
    web_docs = [Document(page_content=s, metadata=data[0].metadata) for s in sections]

    return web_docs


web_docs = web_loader("https://www.uu.se/en/centre/crb/news/archive/2024-09-23-exploring-artificial-consciousness-drawing-inspiration-from-the-human-brain")

# Create a CharacterTextSplitter with specific configuration:
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=800,  # Reduced for better embedding quality
	chunk_overlap=100,  # Added overlap for context continuity
	length_function=len,
	separators=["\n\n", "."],  # Prioritize sentence breaks
	is_separator_regex=False
)

pdf_chunks = text_splitter.split_documents(raw_pdf_docs)
web_chunks = text_splitter.split_documents(web_docs)

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
        print("\nExample chunk:")
        example_doc = docs[min(5, total_chunks-1)]  # Get the 5th chunk or the last one if fewer
        print(f"Content (first 150 chars): {example_doc.page_content[:150]}...")
        print(f"Metadata: {example_doc.metadata}")
        
        # Calculate length distribution
        lengths = [len(doc.page_content) for doc in docs]
        min_len = min(lengths)
        max_len = max(lengths)
        print(f"Min chunk size: {min_len} characters")
        print(f"Max chunk size: {max_len} characters")

# Display stats for both chunk sets
display_document_stats(pdf_chunks, "PDF File")
display_document_stats(web_chunks, "HTML File")