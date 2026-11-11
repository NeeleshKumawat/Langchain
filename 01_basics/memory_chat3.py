# memory_chat.py (production style)

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Step 1: Prompt with MessagesPlaceholder
# MessagesPlaceholder → yahan history inject hogi automatically
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Speak in Hinglish."),
    MessagesPlaceholder(variable_name="history"),  # ← magic yahan hai
    ("human", "{input}")
])

# Step 2: Simple chain
chain = prompt | llm

# Step 3: History store — session wise
# Ek dict jo session_id → ChatMessageHistory map karta hai
store = {}

def get_session_history(session_id: str):
    # Agar session naya hai toh nayi history banao
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Step 4: Chain ko history-aware banao
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,          # history fetch karne ka function
    input_messages_key="input",   # user input ka key
    history_messages_key="history" # prompt mein placeholder ka naam
)

# Step 5: Use karo — session_id dena zaroori hai
config = {"configurable": {"session_id": "nishu_session_1"}}

# Turn 1
r1 = chain_with_history.invoke(
    {"input": "Mera naam Nishu hai aur main Jaipur mein rehta hun"},
    config=config
)
print("AI:", r1.content)

# Turn 2
r2 = chain_with_history.invoke(
    {"input": "Main kahan rehta hun?"},
    config=config
)
print("AI:", r2.content)
# AI: "Tum Jaipur mein rehte ho!"

# Turn 3
r3 = chain_with_history.invoke(
    {"input": "Mera naam kya hai?"},
    config=config
)
print("AI:", r3.content)
# AI: "Tumhara naam Nishu hai!"