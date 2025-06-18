import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

env_path = os.path.join(os.getcwd(), "config", ".env")

_ = load_dotenv(dotenv_path=env_path)


api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

apple_vector = embeddings.embed_query("I love apples.")
fruit_vector = embeddings.embed_query("I enjoy eating fruit.")


print(f"The {embeddings.model} outputs a vector of a fixed dimension:\n")
print(f"Apple vector: {len(apple_vector)} dimensions")
print(f"Fruit vector: {len(apple_vector)} dimensions\n")

for i in range(5):
	print(f"Dimension: {i+1}")
	print(f"Apple vector: {apple_vector[i]}")
	print(f"Fruit vector: {fruit_vector[i]}\n")