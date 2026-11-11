# hello_llm.py

# Step 1: .env se API key load karo
from dotenv import load_dotenv
load_dotenv()  # .env file padh leta hai automatically

# Step 2: LangChain ka OpenAI model import karo
from langchain_groq import ChatGroq

# Step 3: Model banao
# temperature=0 matlab → consistent answers, no randomness
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Step 4: LLM se seedha baat karo
response = llm.invoke("What is 2 + 2?")

# Step 5: Response print karo
print(response.content)
# Output: "4"