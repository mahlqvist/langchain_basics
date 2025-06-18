import os
from langchain_community.document_loaders import UnstructuredPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

pdf_file_path = os.path.join(os.getcwd(), "data", "machine_minds.pdf")


# Load PDF with metadata preservation
loader = UnstructuredPDFLoader(
	pdf_file_path,
	mode="elements",
	infer_table_structure=True
)
raw_pdf_docs = loader.load()

# Preprocess: Filter empty docs
filtered_pdf_docs = [doc for doc in raw_pdf_docs if doc.page_content and len(doc.page_content.strip()) > 5]

web_loader = WebBaseLoader(
    web_path="https://www.uu.se/en/centre/crb/news/archive/2024-09-23-exploring-artificial-consciousness-drawing-inspiration-from-the-human-brain",
    requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0"}},
    default_parser="lxml"
)

# The WebBaseLoader return a single document
raw_web_docs = web_loader.load()

# Split by natural sections (e.g., double newlines)
text = raw_web_docs[0].page_content
sections = [s.strip() for s in text.split("\n\n") if s.strip()]

# Convert to Documents and apply recursive splitting
web_docs = [Document(page_content=s, metadata=raw_web_docs[0].metadata) for s in sections]


# Create a CharacterTextSplitter with specific configuration:
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=600,  # Reduced for better embedding quality
	chunk_overlap=100,  # Added overlap for context continuity
	length_function=len,
	separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],  # Prioritize sentence breaks
	is_separator_regex=False
)

pdf_chunks = text_splitter.split_documents(filtered_pdf_docs)
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