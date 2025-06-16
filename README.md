# Exploring LangChain 🦜 


This is a hands-on repo for learning and experimenting with the core features of **LangChain** 🦜.

This is not a production project, more like a sandbox for exploring what LangChain has to offer!

This repo contains small, focused experiments to help me understand how to use LangChain's tools. Key topics explored:

- **LLMs & Chat Models** – working with OpenAI and other language models.
- **Prompt Templates** – structuring reusable, dynamic prompts.
- **Output Parsers** – extracting structured data from language model responses.
- **Text Splitters** – breaking large documents into chunks for processing.
- **Vector Stores** – storing and searching text using embeddings (e.g. FAISS, ChromaDB).
- **Retrieval** – querying external documents with context-aware answers.
- **Memory** – maintaining conversation history across interactions.
- **Agents & Tools** – building autonomous chains that make decisions and call tools.
- **Chains** – connecting multiple steps together into workflows.

### Project Structure

```
langchain_basics/
│
├── notebooks/         # Jupyter notebooks for quick experiments
├── examples/          # Simple Python scripts by concept
├── prompts/           # Custom prompt templates
├── data/              # Sample input documents
├── requirements.txt   # Python dependencies
└── README.md          # You're here!
```

Create a virtual environment:

```bash
python -m venv env

source env/bin/activate # on Windows: env\Scripts\activate

# for jupyter notebook
pip install ipykernel

# add your virtual env to Jupyter, replace myenv with the name of your venv
python -m ipykernel install --user --name=env

# if you want interactive html-widgets install ipywidgets
pip install ipywidgets

# install the required packages
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your-api-key-here
```

Examples on how it might look:

* `examples/chat_model_basic.py` – Chat with a language model.
* `examples/prompt_template_demo.py` – Build prompts dynamically.
* `examples/output_parser_example.py` – Extract structured output.
* `examples/agent_tool_use.py` – Let an agent decide what to do.
* `examples/vectorstore_retrieval.py` – Search documents with semantic meaning.

I'm building this to understand how LangChain works under the hood and how to use its components in a flexible, modular way. My goal is to master practical tools for building AI-powered apps.