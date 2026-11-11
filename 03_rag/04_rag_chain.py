# 04_rag_chain.py

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Setup (ek baar hota hai) ---

loader = TextLoader("03_rag/data/company_policy.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(chunks, embeddings)

# Retriever — vectorstore ka search interface
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # top 3 chunks laao
)

# --- RAG Prompt ---
prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
""")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# --- Helper function ---
def format_docs(docs):
    # Chunks ko ek string mein join karo
    return "\n\n".join(doc.page_content for doc in docs)

# --- RAG Chain ---
rag_chain = (
    {
        "context": retriever | format_docs,  # question → retriever → chunks → string
        "question": RunnablePassthrough()    # question as-is pass karo
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- Use karo ---
print(rag_chain.invoke("How many sick leaves do employees get?"))
# "Employees get 10 sick leaves per year."

print(rag_chain.invoke("Can I work from home on Monday?"))
# "No, WFH is not allowed on Mondays and Fridays."

print(rag_chain.invoke("What is the salary structure?"))
# "I don't know."  ← document mein nahi tha!