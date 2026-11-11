from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Document load karo
loader = TextLoader("03_rag/data/company_policy.txt")
documents = loader.load()

print("Document loaded:")
print(documents[0].page_content)  # poora text
print("---")
print("Metadata:", documents[0].metadata)
# {"source": "data/company_policy.txt"}


# Step 2: Text splitter banao
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # har chunk max 200 characters
    chunk_overlap=50,     # chunks ke beech 50 chars overlap
)

# Step 3: Document ko chunks mein todo
chunks = splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)