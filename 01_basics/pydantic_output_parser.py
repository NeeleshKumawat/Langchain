from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Step 1: Schema define karo — kya chahiye output mein
class TopicInfo(BaseModel):
    name: str = Field(description="Name of the topic")
    difficulty: str = Field(description="Easy, Medium, or Hard")
    description: str = Field(description="One line description")
    use_cases: List[str] = Field(description="List of use cases")

# Step 2: Parser banao from Pydantic model
parser = PydanticOutputParser(pydantic_object=TopicInfo)

# Step 3: Prompt mein format instructions inject karo
# parser.get_format_instructions() → LLM ko batata hai exact format
prompt = ChatPromptTemplate.from_template("""
Give me information about {topic}.

{format_instructions}
""")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

chain = prompt | llm | parser

# format_instructions parser se aata hai
result = chain.invoke({
    "topic": "RAG pipeline",
    "format_instructions": parser.get_format_instructions()
})

print(result)
# Output: TopicInfo object — not dict, proper Python object!

print(result.name)        # "RAG Pipeline"
print(result.difficulty)  # "Hard"
print(result.use_cases)   # ["document search", "chatbots"]
print(type(result))       # <class '__main__.TopicInfo'>