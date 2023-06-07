# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Bring in streamlit for UI/app interface
import streamlit as st
import os
import streamlit as st
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
# Load environment variables
#Input your OpenAI API key in an .env
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)

# Create and load PDF Loader
loader = PyPDFLoader('Goldman-Sachs-annual-report-2022.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, collection_name='annualreport')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title("Symply Finance V2.5")
st.write("This version allows you to ask questions based on installed annual reports")
st.write("This version will use the 2022 annual report of Goldman Sachs")

st.subheader("You can also ask investment banking questions aside from the Investment questions")

prompt = st.text_input("Please ask your question")
# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 