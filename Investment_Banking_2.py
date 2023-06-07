import os
import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
os.environ['SERPER_API_KEY'] = os.environ['SERPER_API_KEY']

llm = OpenAI(temperature=0.1, verbose=True)
search = GoogleSerperAPIWrapper(api_key=os.getenv('SERPER_API_KEY'))

loader = PyPDFLoader('Goldman-Sachs-annual-report-2022.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, collection_name='annualreport')

vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

st.title("Warren V1.5A")
st.write("This version allows you to ask questions based on installed annual reports")
st.write("This version will use the 2022 annual report of Goldman Sachs")
st.subheader("You can also ask investment banking questions aside from the Investment questions")

prompt = st.text_input("Please ask your question")

def satisfactory(response):
    # Define your own criteria for a satisfactory response
    return len(response) > 20

if prompt:
    response = agent_executor.run(prompt)
    
    if not satisfactory(response):
        search_result = search.run(prompt)
        response = search_result
    
    st.write(response)

    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)
