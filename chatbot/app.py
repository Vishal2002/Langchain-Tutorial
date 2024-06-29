from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langsmith import Client

load_dotenv()

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
client = Client()
tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT"))
callback_manager = CallbackManager([tracer])
## Prompt Template 
prompt = ChatPromptTemplate.from_messages([
    ("human", "You are a helpful assistant. Please respond to the following query: {question}")
])

## Streamlit framework
st.title('Langchain Demo With Gemini')
input_text = st.text_input("Search the topic you want")

# Gemini Pro LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    try:
        response = chain.invoke({'question': input_text})
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")