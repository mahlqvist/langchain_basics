import gradio as gr
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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



with gr.Blocks() as demo:
	response = gr.Textbox(label="AI")
	query = gr.Textbox(label="Human", lines=2, placeholder="Ask anything...")
	send_btn = gr.Button("Send")

	@send_btn.click(inputs=query, outputs=response)
	def get_response(query):
		prompt = ChatPromptTemplate([
			 ("system", "You are an AI Assistant. Answer the User's queries succinctly in one sentence."),
			 ("human", "{usr_query}")
		])
		
		chain = prompt | llm | StrOutputParser()
		
		response = chain.invoke({"usr_query": query})

		return response

demo.launch()