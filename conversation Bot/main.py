import logging
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
import pdfplumber
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

# Load environment variables and setup FastAPI
app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentState:
    def __init__(self, summary=None, last_answer=None):
        self.summary = summary
        self.last_answer = last_answer
        self.asked_questions = []  # Keep a history of asked questions
        self.last_question = None  # Track the last question asked

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize the OpenAI Chat model using LangChain
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4",
    temperature=0.7
)

# Load the sentence transformer model for dynamic vague answer detection
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Precompute embeddings for "move on" intent responses
move_on_responses = ["next question please", "I don’t know", "pass", "skip", "move on"]
move_on_embeddings = model.encode(move_on_responses, convert_to_tensor=True)

# Define the Summarization Chain (Map-Reduce or Refine)
def summarize_resume(text: str, method="map_reduce"):
    documents = [Document(page_content=text)]
    summarize_chain = load_summarize_chain(llm, chain_type=method)
    summary = summarize_chain.invoke(documents)
    return summary

# Function to parse PDF resume using pdfplumber and return text
def parse_resume(pdf_file_data: bytes):
    try:
        logging.info(f"Attempting to parse PDF resume")
        text = ""
        pdf_file = BytesIO(pdf_file_data)
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if not text:
            raise ValueError("No text found in the PDF.")
        logging.info(f"Parsed text length: {len(text)} characters")
        return text
    except Exception as e:
        logging.error(f"Error parsing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

# Generate the initial interview question based on the resume summary
def generate_initial_question(summary: str) -> str:
    logging.info(f"Generating initial question based on summary: {summary}")
    prompt = f"""
    Based on the following candidate summary:
    {summary}
    Generate a first interview question that covers both the candidate's skills and their most recent work experience.
    """
    response = llm.invoke(prompt)
    logging.info(f"First question generated: {response.content.strip()}")
    return response.content.strip()

# Detect if the candidate's response implies they want to move on
def is_move_on_intent(answer: str) -> bool:
    # Encode the candidate's answer
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    
    # Calculate the cosine similarity with the "move on" embeddings
    similarities = util.pytorch_cos_sim(answer_embedding, move_on_embeddings)
    
    # If any similarity is above a certain threshold (e.g., 0.7), it's considered a "move on" intent
    return max(similarities[0]).item() > 0.7

# Generate a clarification question for vague answers
def generate_clarification_question(answer: str, summary: str) -> str:
    logging.info(f"Generating clarification question based on vague answer: {answer}")
    
    # Modify the prompt to generate only one question
    prompt = f"""
    The candidate responded with "{answer}", which seems vague or insufficient. 
    Please generate a **single** direct clarifying question to understand why the candidate couldn't provide more details or gave such a short answer.
    
    The question should be polite and encourage the candidate to elaborate.
    """
    
    response = llm.invoke(prompt)
    question = response.content.strip()  # Get only the content of the response
    logging.info(f"Clarification question generated: {question}")
    return question

# Generate follow-up questions based on the candidate's answer and the resume summary
def generate_follow_up_question(answer: str, summary: str, question: str, asked_questions: list) -> str:
    logging.info(f"Generating follow-up question based on answer: {answer}")
    
    # If the candidate's response suggests they want to move on, skip clarification and ask the next question
    if is_move_on_intent(answer):
        logging.info("Candidate wants to move on. Skipping clarification.")
        
        # Proceed with generating the next follow-up question
        prompt = f"""
        Based on the following resume summary:
        {summary}
        Generate the next interview question focusing on the candidate's skills or experience.
        """
        response = llm.invoke(prompt)
        next_question = response.content.strip()

        # Make sure it's not too similar to previously asked questions
        if not is_question_similar(next_question, asked_questions):
            asked_questions.append(next_question)
            logging.info(f"Next question generated: {next_question}")
            return next_question
        else:
            logging.info(f"Generated question was too similar. Asking another question.")
            return generate_follow_up_question(answer, summary, question, asked_questions)

    # If the candidate's answer isn't an indication to move on, proceed with clarification or follow-up
    if is_answer_vague_dynamic(question, answer):
        logging.info("Candidate's answer is vague or insufficient. Asking for clarification.")
        
        # Generate a clarification question to address the vague response
        clarification_question = generate_clarification_question(answer, summary)
        return clarification_question

    # Generate a follow-up question for valid answers
    prompt = f"""
    Based on the candidate's answer: {answer} and the following resume summary:
    {summary}
    Generate a follow-up interview question that dives deeper into the candidate's skills or experience.
    """
    
    # Generate the follow-up question
    response = llm.invoke(prompt)
    new_question = response.content.strip()

    # Check if the new question is too similar to previously asked questions
    if not is_question_similar(new_question, asked_questions):
        asked_questions.append(new_question)
        logging.info(f"Follow-up question generated: {new_question}")
        return new_question

    logging.warning("Couldn't generate a unique follow-up question after several attempts.")
    return "Please provide more details about your recent experience."

# Helper function to check if the generated question is too similar to previously asked questions
def is_question_similar(new_question: str, asked_questions: list, threshold=0.7) -> bool:
    for question in asked_questions:
        # Encode both the new question and the asked question
        question_embedding = model.encode(question, convert_to_tensor=True)
        new_question_embedding = model.encode(new_question, convert_to_tensor=True)
        
        # Compute the cosine similarity between the new question and each asked question
        similarity = util.pytorch_cos_sim(question_embedding, new_question_embedding)
        
        # If the similarity is higher than the threshold, the questions are too similar
        if similarity.item() > threshold:
            return True  # The question is too similar to one already asked
    
    return False  # The question is not too similar

# Helper function to check if the candidate's answer is vague
def is_answer_vague_dynamic(question: str, answer: str) -> bool:
    question_embedding = model.encode(question, convert_to_tensor=True)
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(question_embedding, answer_embedding)
    
    similarity_threshold = 0.4
    return similarity.item() < similarity_threshold

# Define the StateGraph for the interview process
class InterviewGraph(StateGraph):
    def __init__(self, state: AgentState):
        super().__init__(state)
        self.state = state

        # Define nodes (custom interview flow)
        self.add_node("START_INTERVIEW", self.start_interview)
        self.add_node("ASK_QUESTION", self.ask_question)
        self.add_node("FOLLOW_UP", self.follow_up_question)
        self.add_node("END_INTERVIEW", self.end_interview)

        # Define transitions
        self.add_edge("__start__", "START_INTERVIEW")
        self.add_edge("START_INTERVIEW", "ASK_QUESTION")
        self.add_edge("ASK_QUESTION", "FOLLOW_UP")
        self.add_edge("FOLLOW_UP", "ASK_QUESTION")
        self.add_edge("FOLLOW_UP", "END_INTERVIEW")

    def start_interview(self):
        logging.info("Interview started.")
        return "Let's start the interview!"

    def ask_question(self):
        logging.info("Asking first question.")
        first_question = generate_initial_question(self.state.summary)
        return first_question

    def follow_up_question(self):
        logging.info("Asking follow-up question.")
        answer = self.state.last_answer
        summary = self.state.summary
        question = self.state.last_question  # Last question asked
        follow_up = generate_follow_up_question(answer, summary, question, self.state.asked_questions)
        return follow_up

    def end_interview(self):
        logging.info("Interview ended.")
        return "Thank you for the interview!"

# WebSocket interview session
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        state = AgentState()

        while True:
            data = await websocket.receive_text()
            logging.info(f"Received data: {data}")

            if data.startswith("UPLOAD_RESUME"):
                # Resume file data coming as bytes through WebSocket
                pdf_data = await websocket.receive_bytes()
                resume_text = parse_resume(pdf_data)  # Parse the uploaded resume
                state.summary = summarize_resume(resume_text, method="map_reduce")  # Summarize the resume
                logging.info(f"Resume summary: {state.summary}")

                # Initialize the interview graph
                graph = InterviewGraph(state)
                
                # Start the interview by manually transitioning to the first state
                start_message = graph.start_interview()
                # await websocket.send_text(json.dumps({"message": start_message}))
                
                # Ask the first question by invoking the corresponding method
                first_question = graph.ask_question()
                state.last_question = first_question  # Save the last question asked
                await websocket.send_text(json.dumps({"question": first_question}))

            elif data.startswith("ANSWER"):
                # Save the candidate's answer in the state
                answer = data.split(":")[1].strip()
                logging.info(f"Received answer: {answer}")
                state.last_answer = answer

                # Ask a follow-up question by invoking the corresponding method
                follow_up_question = graph.follow_up_question()
                state.last_question = follow_up_question  # Save the last question asked
                await websocket.send_text(json.dumps({"question": follow_up_question}))

            elif data == "END":
                # End the interview by transitioning to the end state
                end_message = graph.end_interview()
                await websocket.send_text(json.dumps({"message": end_message}))
                break

    except WebSocketDisconnect:
        logging.info("Client disconnected.")
