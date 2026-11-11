from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# History list — yahan sab store hoga
chat_history = []

# --- Turn 1 ---
user_input = "Mera naam Nishu hai"

# User message history mein daalo
chat_history.append(HumanMessage(content=user_input))

# Poori history bhejo LLM ko
response = llm.invoke(chat_history)

# AI response bhi history mein daalo
chat_history.append(AIMessage(content=response.content))

print("AI:", response.content)
# AI: "Nice to meet you, Nishu!"

# --- Turn 2 ---
user_input = "Mera naam kya hai?"

chat_history.append(HumanMessage(content=user_input))

response = llm.invoke(chat_history)

chat_history.append(AIMessage(content=response.content))

print("AI:", response.content)
# AI: "Tumhara naam Nishu hai!"  ← yaad hai ab!