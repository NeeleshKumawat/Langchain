# output_parser.py

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Model
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Prompt
prompt = ChatPromptTemplate.from_template(
    "Tell me one fun fact about {animal}."
)

# Parser — sirf clean string chahiye
parser = StrOutputParser()

# Chain: prompt → llm → parser
chain = prompt | llm | parser

# Invoke
result = chain.invoke({"animal": "octopus"})

print(result)
print(type(result))  # <class 'str'>  ← direct string, no .content needed