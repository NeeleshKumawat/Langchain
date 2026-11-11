from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Template banao with placeholder
# {topic} ek variable hai — baad mein fill karenge
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 2 lines."
)

# Step 2: Model banao
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Step 3: Prompt + LLM ko pipe operator se jodo
# Yeh hai LangChain ka LCEL (LangChain Expression Language)
chain = prompt | llm

# Step 4: Chain chalao, {topic} fill karo
response = chain.invoke({"topic": "machine learning"})

print(response.content)
# Output: "Machine learning ek aisi technology hai..."