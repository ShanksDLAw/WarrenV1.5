import os
import streamlit as st
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain.utilities import GoogleSerperAPIWrapper

# Load environment variables
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
os.environ['SERPER_API_KEY'] = os.environ['SERPER_API_KEY']

llm = LangchainOpenAI(temperature=0.1)
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
st.write("We are your AI-powered Financial Wizard")
st.subheader("Here's some additional information:")
st.write("This app provides general information about investing in financial markets.")
st.write("It's important to do your own research and consider professional advice before making any investment decisions.")

prompt = st.text_input('What would you like to know or explained')

def satisfactory(response):
    # Define your own criteria for a satisfactory response
    return len(response) > 20

if st.button("Submit"):
    if prompt:
        response = agent_executor.run(prompt)
        
        if not satisfactory(response):
            search_result = search.run(prompt)
            response = "Top search results:"
            for i, result in enumerate(search_result[:3], start=1):
                response += f"\n{i}. {result['title']} - {result['url']}"

        st.write(response)

        with st.expander('Document Similarity Search'):
            search = store.similarity_search_with_score(prompt)
            st.write(search[0][0].page_content)
    else:
        st.write("Please enter a question.")
