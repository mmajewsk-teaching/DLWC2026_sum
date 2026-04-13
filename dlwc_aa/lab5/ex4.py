from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SQLiteVec
from langchain_ollama import OllamaEmbeddings, ChatOllama
import sqlite3
import sqlite_vec
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_PATH = "shrek_vec.db"

# Load and split text
loader = ...("shrek.txt")
documents = loader.load()
splitter = ...(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Initialize SQLiteVec (stores both text & vectors)
embedding_model = ...(model="mistral")
connection = sqlite3.connect(DB_PATH, check_same_thread=False)
connection.row_factory = sqlite3.Row
connection.enable_load_extension(True)
sqlite_vec.load(connection)
connection.enable_load_extension(False)
vector_store = ...(table="langchain", connection=..., embedding=..., db_file=...)
vector_store.add_documents(...)

# Create retriever and LLM
retriever = ....as_retriever()
llm = ...(model="mistral")

# Build RAG chain using LCEL (LangChain Expression Language)
prompt = ChatPromptTemplate.from_template(
    "Answer the question based on the following context:\n\n"
    "{context}\n\n"
    "Question: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": ... | format_docs, "question": ...}
    | ...
    | ...
    | ...
)

# Chat function
def chat():
    print("Shrek Chatbot: Ask me anything about Shrek!")
    while True:
        query = ....("You: ")
        if query.lower() in [...]:
            print("Shrek Chatbot: Goodbye!")
            break
        response = rag_chain....(...)
        print(f"Shrek Chatbot: {response}")

if __name__ == "__main__":
    chat()
