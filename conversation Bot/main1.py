import logging
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
from io import BytesIO
import pdfplumber

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

# Initialize the open-source model from Hugging Face (GPT-J or GPT-Neo)
model_name = "EleutherAI/gpt-j-6B"  # Switch to GPT-Neo if preferred
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()

# Initialize SentenceTransformer for vague answer detection
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Precompute embeddings for "move on" intent responses
move_on_responses = ["next question please", "I don’t know", "pass", "skip", "move on"]
move_on_embeddings = sentence_model.encode(move_on_responses, convert_to_tensor=True)

# Function to generate responses using GPT-J or GPT-Neo
def generate_llm_response(prompt: str, max_length: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to summarize resume (using LLM for simplicity here)
def summarize_resume(text: str) -> str:
    prompt = f"Summarize this resume: {text}"
    return generate_llm_response(prompt, max_length=150)

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
    prompt = f"Based on the following summary, generate a first interview question: {summary}"
    return generate_llm_response(prompt)

# Detect if the candidate's response implies they want to move on
def is_move_on_intent(answer: str) -> bool:
    answer_embedding = sentence_model.encode(answer, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(answer_embedding, move_on_embeddings)
    return max(similarities[0]).item() > 0.7

# Generate a clarification question for vague answers
def generate_clarification_question(answer: str) -> str:
    logging.info(f"Generating clarification question based on vague answer: {answer}")
    prompt = f"The candidate responded with '{answer}', which seems vague. Please generate a single polite clarification question."
    return generate_llm_response(prompt)

# Generate follow-up questions based on the candidate's answer and the resume summary
def generate_follow_up_question(answer: str, summary: str, question: str, asked_questions: list) -> str:
    logging.info(f"Generating follow-up question based on answer: {answer}")
    
    # If the candidate's response suggests they want to move on, skip clarification and ask the next question
    if is_move_on_intent(answer):
        logging.info("Candidate wants to move on. Skipping clarification.")
        prompt = f"Based on the following summary, generate the next interview question: {summary}"
        next_question = generate_llm_response(prompt)
        if not is_question_similar(next_question, asked_questions):
            asked_questions.append(next_question)
            return next_question
        return generate_follow_up_question(answer, summary, question, asked_questions)

    # Generate a clarification question if the answer is vague
    if is_answer_vague_dynamic(question, answer):
        return generate_clarification_question(answer)

    # Generate a follow-up question if the answer is valid
    prompt = f"Based on the candidate's answer: {answer} and the resume summary, generate a follow-up interview question."
    return generate_llm_response(prompt)

# Helper function to check if the generated question is too similar to previously asked questions
def is_question_similar(new_question: str, asked_questions: list, threshold=0.7) -> bool:
    for question in asked_questions:
        question_embedding = sentence_model.encode(question, convert_to_tensor=True)
        new_question_embedding = sentence_model.encode(new_question, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(question_embedding, new_question_embedding)
        if similarity.item() > threshold:
            return True
    return False

# Helper function to check if the candidate's answer is vague
def is_answer_vague_dynamic(question: str, answer: str) -> bool:
    question_embedding = sentence_model.encode(question, convert_to_tensor=True)
    answer_embedding = sentence_model.encode(answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(question_embedding, answer_embedding)
    return similarity.item() < 0.4

# Define WebSocket interview session
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
                resume_text = parse_resume(pdf_data)
                state.summary = summarize_resume(resume_text)
                logging.info(f"Resume summary: {state.summary}")

                # Start the interview
                first_question = generate_initial_question(state.summary)
                state.last_question = first_question
                await websocket.send_text(json.dumps({"question": first_question}))

            elif data.startswith("ANSWER"):
                answer = data.split(":")[1].strip()
                logging.info(f"Received answer: {answer}")
                state.last_answer = answer

                # Ask a follow-up question or clarification
                follow_up_question = generate_follow_up_question(answer, state.summary, state.last_question, state.asked_questions)
                state.last_question = follow_up_question
                await websocket.send_text(json.dumps({"question": follow_up_question}))

            elif data == "END":
                await websocket.send_text(json.dumps({"message": "Interview ended."}))
                break

    except WebSocketDisconnect:
        logging.info("Client disconnected.")
