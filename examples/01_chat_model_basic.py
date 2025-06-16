import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

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

prompts = {"zero_shot_prompt": "What are the key components of a neural network?",
		   "few_shot_prompt": """
Extract the animal and its color from each sentence.

Example 1:
Sentence: "The black dog ran across the yard."
Animal: dog
Color: black

Example 2:
Sentence: "A white cat sat on the windowsill."
Animal: cat
Color: white

Example 3:
Sentence: "A brown horse galloped through the field."
Animal: horse
Color: brown

Now you try:
Sentence: "The gray elephant splashed water with its trunk."
Animal:
Color:
""",
		"chain_of_thought_promp": """
When I was 6, my sister was half of my age. Now I am 70, what age is my sister?

Break down each step of your calculation.
""",
		"self_consistancy_prompt": """
Lily has 4 times as many apples as Tom. Together, they have 50 apples. How many apples does Tom have?

Provide three independent calculations and explanations, then determine the most consistent result.
"""
}


for name, prompt in prompts.items():
	response = llm.invoke(prompt)
	name = " ".join(name.split("_")).title()
	print(f"\nType: {name}\n")
	print(f"AI Response:\n\n{response.content}\n\n")