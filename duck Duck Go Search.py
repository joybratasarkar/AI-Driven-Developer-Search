import requests
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
import os

openai_api_key = os.getenv("OPENAI_API_KEY")

# Function to search DuckDuckGo
def search_duckduckgo(query):
    url = f'https://api.duckduckgo.com/?q={query}&format=json'
    response = requests.get(url)
    data = response.json()

    # Extract the relevant fields from the DuckDuckGo response
    answer = data.get('AbstractText', 'No relevant information found on DuckDuckGo.')
    return answer

# Define the DuckDuckGo search tool
duckduckgo_tool = Tool(
    name="DuckDuckGoSearch",
    func=search_duckduckgo,
    description="Searches DuckDuckGo for relevant information."
)

# Initialize an LLM (e.g., OpenAI GPT)
llm = ChatOpenAI(temperature=0.5, model_name="gpt-4o")

# Initialize the agent with the DuckDuckGo tool
agent = initialize_agent(
    tools=[duckduckgo_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ask the agent a question
try:
    response = agent.invoke("Search for the total amount of funds raised or seed funding received by Supersourcing in recent news or financial reports. Include details about any investment rounds or funding announcements from 2023 onwards.")
except Exception as e:
    print(f"An error occurred: {e}")
print(response)
