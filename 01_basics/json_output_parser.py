# output_parser.py mein add karo (ya alag file)

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Prompt mein explicitly bolo — JSON format mein do
prompt = ChatPromptTemplate.from_template("""
Give me information about {topic} as a JSON object.

Return ONLY this JSON structure, nothing else:
{{
  "name": "topic name",
  "difficulty": "Easy/Medium/Hard",
  "description": "one line description",
  "use_cases": ["use case 1", "use case 2"]
}}
""")

parser = JsonOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"topic": "neural networks"})

print(result)
# Output:
# {
#   "name": "Neural Networks",
#   "difficulty": "Hard",
#   "description": "...",
#   "use_cases": ["image recognition", "NLP"]
# }

print(type(result))        # <class 'dict'>
print(result["name"])      # "Neural Networks"
print(result["use_cases"]) # ["image recognition", "NLP"]