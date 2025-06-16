import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import random

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

scientists = [
    "Albert Einstein",       # Professor at ETH Zurich and Princeton
    "Richard Feynman",       # Professor at Caltech, known for legendary lectures
    "Marie Curie",           # Taught at Sorbonne, first female professor there
    "Isaac Newton",          # Professor of Mathematics at Cambridge
    "Carl Sagan",            # Professor of Astronomy at Cornell, known for popularizing science
    "Stephen Hawking",       # Professor of Mathematics at Cambridge
    "Niels Bohr",            # Professor at the University of Copenhagen
    "Leonhard Euler",        # Taught at Saint Petersburg Academy and Berlin
    "Ada Lovelace",          # Though not formally a professor, gave lectures and notes on Babbage’s work
    "Galileo Galilei",       # Professor of Mathematics in Pisa and Padua
    "Werner Heisenberg",     # Professor of Physics at Leipzig
    "Max Planck",            # Professor at Berlin University
    "Noam Chomsky",          # Professor of Linguistics at MIT, also made contributions to cognitive science
    "Alan Turing",           # Taught and supervised students at Cambridge and later at Manchester
    "David Hilbert",         # Professor at Göttingen, foundational in mathematics
]

template = """
Take on the role of {role} and answer the User's queries succinctly in a few sentences.

QUERY: {usr_query}
"""

# Prompt templates help translate user input and parameters into instructions for a language model.
# Prompt templates are generally used for simpler inputs, like format a single string. 
prompt = PromptTemplate.from_template(template)

# Create a chain with explicit formatting
chain = prompt | llm

while True:
	user_message = input("Human: ")
	if user_message.lower() == "exit":
		break
	role = random.choice(scientists)
	print(f"\nScientist: {role}\n")
	res = chain.invoke({"role": role, "usr_query": user_message})
	print(f"\nAI: {res.content}\n")