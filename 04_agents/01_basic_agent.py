from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

# --- Tools banao ---

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. 
    Use this for any weather related questions."""
    data = {
        "jaipur": "32°C, Sunny",
        "delhi": "28°C, Cloudy",
        "mumbai": "30°C, Humid"
    }
    return data.get(city.lower(), "City not found")

@tool
def get_population(city: str) -> str:
    """Get population of a city.
    Use this when asked about population or number of people."""
    data = {
        "jaipur": "40 lakh",
        "delhi": "2 crore",
        "mumbai": "2.1 crore"
    }
    return data.get(city.lower(), "Data not available")

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    Use this for any calculations. 
    Example: '100 * 32' or '500 / 4'"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression"

# --- Agent banao ---
# create_agent = LangChain 1.0 ka naya way
agent = create_agent(
    model="groq:openai/gpt-oss-120b",           # model string
    tools=[get_weather,               # tools ki list
           get_population, 
           calculator],
    system_prompt="""You are a helpful assistant.
    Answer questions about weather, population, and calculations.
    Always use tools when needed. Respond in Hinglish."""
)

# --- Use karo ---
result = agent.invoke({
    "messages": [{"role": "user", "content": "Delhi aur Jaipur ki population ka sum kya hai?"}]
})
print("result ------------> ", result)