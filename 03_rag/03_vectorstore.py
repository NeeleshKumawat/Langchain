# 03_vectorstore.py

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Step 1: Load + chunk (same as before)
loader = TextLoader("03_rag/data/company_policy.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# Step 2: Embeddings model
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Vector store banao
# Yeh andar:
# 1. Har chunk ka embedding banata hai
# 2. Chunks + embeddings store karta hai
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # disk pe save hoga
)

print("Vector store ready!")
print(f"Total vectors stored: {vectorstore._collection.count()}")

# Step 4: Similarity search karo
query = "How many sick leaves?"
results = vectorstore.similarity_search(query, k=2)  # top 2 chunks

print(f"\nQuery: {query}")
print("\nRelevant chunks:")
for r in results:
    print("---")
    print(r.page_content)