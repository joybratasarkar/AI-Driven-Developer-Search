import logging
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
import PyPDF2
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load OpenAI API key from environment variable for security
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize the OpenAI Chat model using LangChain
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4",
    temperature=0.7
)

# Session data per WebSocket connection
clients_data = {}

# Generate a fixed number of questions (between 4 and 8)
def generate_questions_tool(resume: str, num_questions: int = 5) -> list:
    logging.info("Generating questions for resume...")
    try:
        prompt = f"""
        You are an interviewer. Here is a candidate's resume:
        {resume}
        Generate {num_questions} questions that are relevant to the candidate's skills and experience.
        """
        response = llm.invoke(prompt)
        questions = response.content.split("\n")
        questions = [q.strip() for q in questions if q.strip()]
        return questions[:num_questions]  # Return only the desired number of questions
    except Exception as e:
        logging.error(f"Error generating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

# Parse PDF resume
def parse_resume(pdf_file: UploadFile):
    try:
        logging.info(f"Attempting to parse file: {pdf_file.filename}")
        if pdf_file.filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(pdf_file.file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            if not text:
                raise ValueError("No text found in the PDF.")
            logging.info(f"Parsed text length: {len(text)} characters")
            return text
        else:
            raise ValueError(f"Unsupported file type: {pdf_file.filename}")
    except Exception as e:
        logging.error(f"Error parsing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

@app.post("/upload_resume/")
async def upload_resume(file: UploadFile):
    resume_text = parse_resume(file)
    return {"message": "Resume uploaded successfully"}

# WebSocket endpoint to handle interview session
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients_data[websocket] = {"questions": [], "answers": [], "current_question_index": 0}

    try:
        while True:
            data = await websocket.receive_text()

            if data.startswith("UPLOAD_RESUME"):
                resume_text = "Sample resume text for testing"  # Simulate resume parsing for now
                questions = generate_questions_tool(resume_text)
                clients_data[websocket]["questions"] = questions

                # Send the first question
                first_question = clients_data[websocket]["questions"][0]
                await websocket.send_text(json.dumps({"question": first_question}))

            elif data.startswith("ANSWER"):
                answer = data.split(":")[1]
                clients_data[websocket]["answers"].append(answer)

                # Move to the next question
                index = clients_data[websocket]["current_question_index"]
                index += 1
                clients_data[websocket]["current_question_index"] = index

                if index < len(clients_data[websocket]["questions"]):
                    next_question = clients_data[websocket]["questions"][index]
                    await websocket.send_text(json.dumps({"question": next_question}))
                else:
                    # No more questions, send interview completion message
                    await websocket.send_text(json.dumps({"message": "Interview complete. Thank you!"}))
                    break

    except WebSocketDisconnect:
        logging.info("Client disconnected.")
        del clients_data[websocket]
