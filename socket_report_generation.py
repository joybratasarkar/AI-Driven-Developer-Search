from fastapi import FastAPI, WebSocket, WebSocketDisconnect,HTTPException,Query
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
import requests
import os
import asyncio

import json
from schema import ReportSchema, report_collection
from pymongo import MongoClient
from pymongo.collection import Collection
from pydantic import BaseModel
from typing import List,Optional

# Ensure your OpenAI API key is set in the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Store progress for each client connection
client_progress = {}

# Function to search DuckDuckGo
def search_duckduckgo(query):
    # print(f"Searching DuckDuckGo for query: {query}")  # Debug statement
    url = f'https://api.duckduckgo.com/?q={query}&format=json'
    response = requests.get(url)
    data = response.json()
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
print("Initializing agent with DuckDuckGo tool...")  # Debug statement
agent = initialize_agent(
    tools=[duckduckgo_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Function to use LLM to summarize text using Refine Approach
def refine_summary(text, refinement_steps=2):
    # print(f"Refining summary for text: {text}")  # Debug statement
    summary = llm.invoke(f"Summarize the following text:\n\n{text}")
    
    for _ in range(refinement_steps):
        summary = llm.invoke(f"Refine the following summary to make it more concise and clear:\n\n{summary.content}")
    
    return summary.content

# Function to use LLM to evaluate the similarity
def evaluate_answer_with_llm(candidate_answer, correct_answer):
    # print(f"Evaluating answer with LLM: Candidate Answer: {candidate_answer}, Correct Answer: {correct_answer}")  # Debug statement
    prompt = (
        f"You are an AI assistant helping recruiters evaluate a candidate's fit for a specific role. Consider the following aspects when evaluating the candidate's answer against the correct answer:\n\n"
        f"**1. Technical Skills**\n"
        f"- **Relevance to Role**: How well does the candidate's answer demonstrate skills that are directly relevant to the job requirements?\n"
        f"- **Depth of Knowledge**: Does the candidate's answer reflect a deep understanding and expertise in the subject matter?\n"
        f"- **Problem-Solving Ability**: How effectively does the candidate approach and solve the problem presented in the question?\n\n"
        f"**2. Experience**\n"
        f"- **Relevant Work Experience**: How well does the answer showcase experience that is applicable to the job?\n"
        f"- **Adaptability and Learning**: Does the answer indicate that the candidate can adapt to new challenges and learn new skills?\n\n"
        f"**3. Soft Skills**\n"
        f"- **Communication Skills**: Is the answer clear, well-structured, and easy to understand?\n"
        f"- **Teamwork and Collaboration**: Does the answer suggest the candidate's ability to work effectively within a team?\n"
        f"- **Cultural Fit**: How well does the candidate's answer align with the company's values and culture?\n\n"
        f"**4. Behavioral Competencies**\n"
        f"- **Leadership and Initiative**: Does the answer demonstrate leadership qualities or the ability to take initiative?\n"
        f"- **Work Ethic and Attitude**: What does the answer reveal about the candidate's work ethic and attitude towards their work?\n"
        f"- **Decision-Making Skills**: How well does the candidate make decisions or justify their approach in the answer?\n\n"
        f"Candidate's Answer: {candidate_answer}\n\n"
        f"Correct Answer: {correct_answer}\n\n"
        f"Based on the above criteria, provide a detailed assessment of the candidate's answer and assign a score between 0 and 100 for each category. Additionally, provide an overall score indicating how well the candidate's answer meets the expectations for the role and a brief explanation for each score to highlight the strengths and areas for improvement."
    )

    response = llm.invoke(prompt)
    return response.content

# Function to process each question-answer pair
async def process_question_answer(item, websocket: WebSocket, total_items: int, current_index: int, client_id: str):
    question = item['question']
    candidate_answer = item['answer']

    # print(f"Processing question: {question}")  # Debug statement
    try:
        # Retrieve the correct answer dynamically using the agent
        result = agent.invoke(f"Search the internet to answer: {question}")
        correct_answer = result['output']
        # print(f"Correct answer retrieved from the internet for question '{question}': {correct_answer}")

        # Summarize the correct answer using Refine Approach
        summarized_correct_answer = refine_summary(correct_answer)
        # print(f"Summarized correct answer: {summarized_correct_answer}")

        # Use LLM to evaluate the candidate's answer
        evaluation_score = evaluate_answer_with_llm(candidate_answer, summarized_correct_answer)
        # print(f"The candidate's answer correctness score for question '{question}' is:\n{evaluation_score}")

        # Calculate remaining progress
        remaining_percentage = 100 - ((current_index + 1) / total_items) * 100
        client_progress[client_id] = remaining_percentage  # Update progress
        # print(f"Remaining progress: {remaining_percentage}%")

        # Construct the result in the required format
        evaluation_text = (
            f"Summarized correct answer: {summarized_correct_answer}\n"
            f"The candidate's answer correctness score for question '{question}' is: {evaluation_score}\n"
        )

        return {
            "question": question,
            "candidate_answer": candidate_answer,
            "evaluation": evaluation_text
        }

    except Exception as e:
        print(f"An error occurred for question '{question}': {e}")
        return {
            "question": question,
            "candidate_answer": candidate_answer,
            "evaluation": f"Error: {str(e)}"
        }
def save_report(report_data):
    try:
        # Transform the data into the desired format
        document = {
            'projectId': report_data['engineer_id'],
            'results': [
                {
                    'question': result['question'],
                    'candidate_answer': result['candidate_answer'],
                    'evaluation': result['evaluation']
                }
                for result in report_data['results']
            ]
        }
        
        print('Formatted document to save:', document)

        # Insert report into MongoDB
        inserted_id = report_collection.insert_one(document).inserted_id

        print("Report saved successfully.")
        return inserted_id
    except Exception as e:
        print(f"Error saving report to MongoDB: {e}")
        return None

# WebSocket connection handler for report generation
@app.websocket("/reportGeneration1")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket connection established")  # Debug statement
    await websocket.accept()
    client_id = websocket.client.host  # Using the client's IP as a unique identifier
    try:
        while True:
            # Receive conversation history from the client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            # print('request_data',request_data)
            conversation_history = request_data.get('conversation_history', [])
            engineer_id = request_data.get('engineer_id', '')

            # print(f"Received conversation history: {conversation_history}")  # Debug statement
            
            total_items = len(conversation_history)
            results = []

            for idx, item in enumerate(conversation_history):
                result = await process_question_answer(item, websocket, total_items, idx, client_id)
                results.append(result)
            # print('engineer_id',engineer_id)
            print('results---------------------------',results)
            # Send back the structured results to the client
            results_str = json.dumps(results)  # Convert list to JSON string

            report_data = {
                "results": results,  
                "engineer_id":engineer_id
            }
            
            # Save the report to MongoDB
            save_report(report_data)
            await websocket.send_json({'results': results,'engineer_id':engineer_id})
            # print(f"Sent results back to client: {results}")  # Debug statement
    
    except WebSocketDisconnect:
        print("Client disconnected")
        if client_id in client_progress:
            del client_progress[client_id]  # Clean up progress tracking
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        await websocket.close(code=1011, reason='Internal server error')

# WebSocket connection handler for progress tracking
@app.websocket("/progress")
async def progress_endpoint(websocket: WebSocket):
    print("Progress WebSocket connection established")  # Debug statement
    await websocket.accept()
    client_id = websocket.client.host  # Using the client's IP as a unique identifier
    try:
        while True:
            # Send the remaining progress to the client
            remaining_percentage = client_progress.get(client_id, 100)
            await websocket.send_json({"remaining_progress": remaining_percentage})
            await asyncio.sleep(1)  # Send progress updates every second
    except WebSocketDisconnect:
        print("Client disconnected from progress endpoint")
    except Exception as e:
        print(f"An unexpected error occurred in progress endpoint: {e}")
        await websocket.close(code=1011, reason='Internal server error')



class Report(BaseModel):
    projectId: str
    results: List[dict]




@app.get("/reports", response_model=List[Report])
async def get_reports(projectId: Optional[str] = Query(None, description="The projectId to filter reports by")):
    """
    Fetch reports from MongoDB. If a projectId is provided, filter by that projectId.
    """
    try:
        query = {}
        if projectId:
            query['projectId'] = projectId
        
        # Fetch documents from the 'reports' collection based on the query
        reports = list(report_collection.find(query))
        
        # Convert MongoDB documents to Python dicts and remove ObjectId
        formatted_reports = [{**report, '_id': str(report['_id'])} for report in reports]
        
        return formatted_reports
    
    except Exception as e:
        print(f"An error occurred while fetching reports: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
if __name__ == "__main__":
    print("Starting FastAPI server...")  # Debug statement
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
