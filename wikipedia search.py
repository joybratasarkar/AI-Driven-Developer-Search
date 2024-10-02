import wikipediaapi
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
import requests

# Initialize Wikipedia API with a proper user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyOpenSourceTool/1.0 (https://github.com/myopensource/tool/issues)'
)

# Function to search Wikipedia
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
llm = OpenAI()

# Initialize the agent with the Wikipedia tool
agent = initialize_agent(
    tools=[duckduckgo_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ask the agent a question
response = agent.run("FInd  tcs  is a  product based or service based company")
print(response)
