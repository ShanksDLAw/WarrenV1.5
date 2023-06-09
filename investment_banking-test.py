import os
# Import things that are needed generically
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper, GoogleSearchAPIWrapper
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI as LangchainOpenAI
from langchain import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

os.environ["GOOGLE_CSE_ID"]
os.environ["GOOGLE_API_KEY"]



# Load environment variables
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# os.environ['SERPER_API_KEY'] = os.environ['SERPER_API_KEY']

llm = LangchainOpenAI(temperature=0.1)
# search = GoogleSerperAPIWrapper(api_key=os.getenv('SERPER_API_KEY'))

loader = PyPDFLoader('Goldman-Sachs-annual-report-2022.pdf')
pages = loader.load_and_split()
embeddings = OpenAIEmbeddings()
store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')

annual_report = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=store.as_retriever(), verbose=True)



search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name= "Annual Report",
        func=annual_report.run,
        description="Search the annual report of a company",
    ),

    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


# Get user input
user_input = input("Please enter your question: ")
conversation = [
    SystemMessage(content="You are a friendly Stock/Financial analyst that explains financial news and stock reports easily into digestible bits"),
    HumanMessage(content=user_input)
]

# Process the user input
result = agent.run(conversation)
print(result)