# multiple_tools.py

from dotenv import load_dotenv
load_dotenv()

from langchain_groq  import ChatGroq
from langchain_core.tools import tool

# Tool 1: Calculator
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# Tool 2: String tool
@tool
def make_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()

# Tool 3: Fake weather (real mein API call hogi)
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Real app mein yahan OpenWeatherMap API call hogi
    weather_data = {
        "jaipur": "32°C, Sunny",
        "delhi": "28°C, Cloudy",
        "mumbai": "30°C, Humid"
    }
    return weather_data.get(city.lower(), "Weather data not available")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Saare tools ek saath do
llm_with_tools = llm.bind_tools([multiply, make_uppercase, get_weather])

# Tools ka map banao — naam se tool dhundhne ke liye
tools_map = {
    "multiply": multiply,
    "make_uppercase": make_uppercase,
    "get_weather": get_weather
}

def run_with_tools(user_question: str):
    from langchain_core.messages import HumanMessage, ToolMessage

    messages = [HumanMessage(user_question)]

    # LLM se poochho
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Agar tool call hai toh run karo
    while response.tool_calls:
        for tool_call in response.tool_calls:
            # Tool naam se dhundho
            selected_tool = tools_map[tool_call["name"]]

            # Tool run karo
            result = selected_tool.invoke(tool_call["args"])

            # Result history mein add karo
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))

        # LLM ko updated history bhejo
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    return response.content

# Test karo
print(run_with_tools("Jaipur ka weather kya hai?"))
# "Jaipur mein aaj 32°C hai aur sunny hai!"

print(run_with_tools("'hello world' ko uppercase mein karo"))
# "HELLO WORLD"

print(run_with_tools("456 aur 789 ko multiply karo"))
# "456 aur 789 ka product 359,784 hai"