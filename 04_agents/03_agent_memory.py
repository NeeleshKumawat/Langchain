from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    data = {"jaipur": "32°C Sunny", "delhi": "28°C Cloudy"}
    return data.get(city.lower(), "Not found")

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=[get_weather],
    system_prompt="You are helpful. Remember conversation history."
)

# Memory — manually track karo
chat_history = []

def chat(user_input: str):
    # History + current message
    messages = chat_history + [
        {"role": "user", "content": user_input}
    ]

    print("messages-----------> ", messages)
    
    result = agent.invoke({"messages": messages})

    final_answer = result["messages"][-1].content
    
    # History update karo
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": final_answer})
    
    return result

# Conversation
print(chat("Mera naam Nishu hai"))
print(chat("Jaipur ka weather?"))
print(chat("Mera naam kya hai?"))  # ← yaad rahega!