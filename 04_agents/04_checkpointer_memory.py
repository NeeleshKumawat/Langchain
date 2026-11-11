from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    data = {"jaipur": "32°C Sunny", "delhi": "28°C Cloudy"}
    return data.get(city.lower(), "Not found")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are helpful.",
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "nishu_chat_1"}}

def chat(user_input: str):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )
    return result["messages"][-1].content

print(chat("Mera naam Nishu hai"))
print(chat("Jaipur ka weather?"))
print(chat("Mera naam kya hai?"))  # ← agent khud yaad rakhega