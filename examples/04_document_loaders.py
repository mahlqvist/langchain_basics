"""
In order to use document loaders:

pip install -qU langchain-community pypdf beautifulsoup4 lxml

pypdf is for PyPDFLoader
bs4 is for WebBaseLoader
lxml if you want a better html parser
"""
import os
from dotenv import load_dotenv
env_path = os.path.join(os.getcwd(), "config", ".env")
_ = load_dotenv(dotenv_path=env_path)
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
import re
from datetime import datetime as dt

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



# Using WebBaseLoader
def web_loader(web_url):
    web_loader = WebBaseLoader(web_url)
    web_loader.default_parser = "lxml"
    web_loader.requests_kwargs = {"headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}}
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
    return web_data




def main():
    run = True
    while run:
        ans = input("Load [web] or [pdf]: ")
        if "exit" in ans.lower():
            run = False
            print("Goodbye!")
            continue
        elif "web" in ans.lower():
            web_data = web_loader("https://www.uu.se/en/centre/crb/news/archive/2024-09-23-exploring-artificial-consciousness-drawing-inspiration-from-the-human-brain")

            print(f"\nSOURCE: {web_data['metadata']['source']}\n")
            print(f"CONTENT: {web_data['content'][:100]}\n\n")
        elif "pdf" in ans.lower():
            pdf_file_path = os.path.join(os.getcwd(), "data", "machine_minds.pdf")
            documents = document_loader(pdf_file_path, "Machine Minds: The Blueprint of Artificial Consciousness", "Sidharta Chatterjee", "2024-06-21")

            for i in range(2):
                print(f"\nDocument: {i+1}\n")
                print(f"Metadata: {documents[i].metadata}\n")
                print(f"Content: {documents[i].page_content[:100]}...\n\n")
        else: 
            print("Please pick web or pdf...")
            continue


if __name__ == "__main__":
    main()
    
