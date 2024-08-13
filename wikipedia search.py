import wikipediaapi
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI

# Initialize Wikipedia API with a proper user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyOpenSourceTool/1.0 (https://github.com/myopensource/tool/issues)'
)

# Function to search Wikipedia
def search_wikipedia(query):
    # Search for the page in Wikipedia
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary
    else:
        return "No relevant information found on Wikipedia."

# Define the Wikipedia search tool
wikipedia_tool = Tool(
    name="WikipediaSearch",
    func=search_wikipedia,
    description="Searches Wikipedia for relevant information."
)

# Initialize an LLM (e.g., OpenAI GPT)
llm = OpenAI()

# Initialize the agent with the Wikipedia tool
agent = initialize_agent(
    tools=[wikipedia_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ask the agent a question
response = agent.run(" What is the purpose of a TypeScript interface and how do you use it?")
print(response)
