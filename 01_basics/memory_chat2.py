from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# SystemMessage → AI ko role deta hai, yeh HAMESHA pehle aata hai
system = SystemMessage(content="""
You are a helpful assistant who speaks in Hinglish.
Always be friendly and remember what the user tells you.
""")

chat_history = [system]   # system message se shuru

print("Chat shuru karo! 'quit' likhke band karo.\n")

while True:
    user_input = input("Tu: ")

    if user_input.lower() == "quit":
        break

    # User message add karo
    chat_history.append(HumanMessage(content=user_input))

    # Poori history bhejo
    response = llm.invoke(chat_history)

    # AI response add karo
    chat_history.append(AIMessage(content=response.content))

    print(f"AI: {response.content}\n")