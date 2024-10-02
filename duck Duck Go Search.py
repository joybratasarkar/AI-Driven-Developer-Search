import requests
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
import os

# Ensure your OpenAI API key is set in the environment variables
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
llm = ChatOpenAI(temperature=0.5, model_name="gpt-4")

# Initialize the agent with the DuckDuckGo tool
agent = initialize_agent(
    tools=[duckduckgo_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Function to use LLM to summarize text using Refine Approach
def refine_summary(text, refinement_steps=2):
    summary = llm.invoke(f"Summarize the following text:\n\n{text}")
    
    for _ in range(refinement_steps):
        summary = llm.invoke(f"Refine the following summary to make it more concise and clear:\n\n{summary.content}")
    
    return summary.content

# Function to use LLM to evaluate the similarity
def evaluate_answer_with_llm(candidate_answer, correct_answer):
    prompt = (
        f"Evaluate the following candidate answer based on the correct answer provided.\n\n"
        f"Candidate Answer: {candidate_answer}\n\n"
        f"Correct Answer: {correct_answer}\n\n"
        f"Please provide a score between 0 and 100 indicating how correct the candidate's answer is."
    )
    
    # Use invoke method instead of __call__ to avoid deprecation warning
    response = llm.invoke(prompt)
    return response.content

# Candidate answer input
candidate_answer = ("Node.js (Node) is an Open Source, cross-platform runtime environment for executing JavaScript code. "
                    "Node is used extensively for server-side programming, making it possible for developers to use JavaScript "
                    "for client-side and server-side code without needing to learn an additional language. Node is sometimes referred "
                    "to as a programming language or software development framework, but neither is true; it is strictly a JavaScript runtime."
                    "Node incorporates the V8 JavaScript engine, the same one used in Google Chrome and other browsers. It is written in C++ "
                    "and can run on macOS, Linux, Windows and other systems. The engine parses and executes JavaScript code. It can operate "
                    "independently of a browser environment, either embedded in a C++ application or implemented as a standalone program. "
                    "The V8 engine compiles JavaScript internally, using just-in-time (JIT) processes to speed up execution."
                    "The following code is an example of a simple JavaScript file (server_tst.js) written for the Node environment. The script "
                    "starts by loading the Node.js Hypertext Transfer Protocol (HTTP) module. The module includes a variety of classes and methods "
                    "for implementing an HTTP server.")

# Ask the agent to get the correct answer from the internet
try:
    # Retrieve the correct answer dynamically using the agent
    result = agent.invoke("What is Node.js?")
    
    # Extract the 'output' part of the result
    correct_answer = result['output']
    print(f"Correct answer retrieved from the internet: {correct_answer}")

    # Summarize the correct answer using Refine Approach
    summarized_correct_answer = refine_summary(correct_answer)
    print(f"Summarized correct answer: {summarized_correct_answer}")

    # Use LLM to evaluate the candidate's answer
    evaluation_score = evaluate_answer_with_llm(candidate_answer, summarized_correct_answer)
    print(f"The candidate's answer correctness score is: {evaluation_score}%")

except Exception as e:
    print(f"An error occurred: {e}")

{'conversation_history': [{'question': '- How do you tailor your communication style when interfacing with different stakeholders?', 'answer': "Actually, I don't."}, {'question': '- Can you provide an example of a challenging situation where your communication skills were instrumental in reaching a positive outcome?', 'answer': 'Know.'}]}