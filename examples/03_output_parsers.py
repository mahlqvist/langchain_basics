import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


env_path = os.path.join(os.getcwd(), "config", ".env")

_ = load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key: 
	raise ValueError("API key missing")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3,
    top_p=0.9,
)

template = """
You are a film scholar and expert in movies. 

Generate information about "{movie_name}" in JSON format.

Return ONLY a JSON object with no text before or after. The JSON must have these keys:

Output valid JSON in exactly this format:
{{
	"title": "movie title",
  	"director": "director name",
  	"year": year as number,
  	"genre": "movie genre"
}}

If any field doesn't apply or lacks sufficient information, use "N/A"

Your entire response must be valid JSON. Do not include any text outside the JSON object in your response.
"""

prompt = PromptTemplate.from_template(template)

# combined chain using LangChain Expression Language (LCEL)
chain = prompt | llm | JsonOutputParser()

movie_name = "The Matrix"
result = chain.invoke({
    "movie_name": movie_name,
})

print(f"Type: {type(result)}")
print(f"Title: {result['title']}")
print(f"Genre: {result['genre']}")