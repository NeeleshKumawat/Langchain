from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent

# --- RAG Setup ---

# 1. Document load + chunk
loader = TextLoader("03_rag/data/company_policy.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 2. Vector store
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- RAG ko Tool banao ---
@tool
def search_company_policy(query: str) -> str:
    """Search company HR policy documents.
    Use this for questions about leave, WFH, holidays, 
    salary, or any company policy related questions."""
    
    # Retriever se chunks dhundho
    docs = retriever.invoke(query)
    
    # Chunks ko string mein join karo
    return "\n\n".join(doc.page_content for doc in docs)

# Normal tool bhi saath mein
@tool
def get_current_date() -> str:
    """Get today's date. Use when asked about current date or time."""
    from datetime import date
    return str(date.today())

# --- Agent ---
agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=[search_company_policy, get_current_date],
    system_prompt="""You are an HR assistant for Acme Corp.
    Answer employee questions about company policies.
    Always search the policy documents before answering.
    If information is not in documents, say 'Policy not defined'."""
)

# --- Test karo ---
questions = [
    "Mujhe kitne sick leaves milte hain?",
    "Kya main Monday ko WFH kar sakta hun?",
    "Aaj ki date kya hai?",
    "Stock options policy kya hai?"
]

for q in questions:
    print(f"\nQ: {q}")
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    print(f"A: {result}")
    print("─" * 50)