from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

# Embedding model banao
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# Ek sentence ko vector mein convert karo
text = "How many annual leaves do employees get?"
vector = embeddings.embed_query(text)

print(f"Vector length: {len(vector)}")
# Vector length: 1536

print(f"First 5 values: {vector[:5]}")
# [-0.023, 0.041, -0.018, 0.092, -0.055]