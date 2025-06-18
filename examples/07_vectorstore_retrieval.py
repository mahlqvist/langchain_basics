"""
pip install chromadb
"""
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import chromadb


env_path = os.path.join(os.getcwd(), "config", ".env")

_ = load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# Connect to ChromaDB, our vector database, where we store and search embeddings.
# We're using persistent storage so data is saved between runs.
vector_db = chromadb.PersistentClient(path="./chroma_test_storage")

# A collection is like a table in a traditional database—it holds related data.
collection = vector_db.get_or_create_collection(name="test_collection")

# These sentences will be stored in our vector database for future searches.
documents = [
    "LangChain is the best framework!", # A fact? An opinion?
    "The sky is blue.",                 # A simple, objective statement.
    "Fedora is just awesome."           # Biased? Maybe, but it's in our dataset!
]

# Each document is transformed into a vector that represents its meaning.
document_vectors = embeddings.embed_documents(documents)

# Unique IDs for each document
document_ids = ["1", "2", "3"]

# Store documents and their embeddings in ChromaDB
collection.upsert(
    documents=documents,
    embeddings=document_vectors,
    ids=document_ids
)

usr_query = "What's Fedora?"

# Convert the query into an embedding
query_vector = embeddings.embed_query(usr_query)

# Retrieve the collection (ensuring we're querying the right dataset)
collection = vector_db.get_collection("test_collection")

# Search for the closest match using vector similarity
results = collection.query(
    query_embeddings=[query_vector],
    n_results=1,  # We only want the single best match
    include=["documents"]  # Return the matching document, not just the ID
)

# Since our database understands meaning, it should return "Fedora is just awesome."
print("User query:", usr_query)
print("Search result:", results['documents'][0][0])