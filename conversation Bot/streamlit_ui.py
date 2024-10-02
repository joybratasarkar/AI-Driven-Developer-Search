import logging
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
import PyPDF2
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import os

app = FastAPI()
load_dotenv()

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

# Store session data for WebSocket clients
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
        questions = response.content.split("\n")  # Split the response into questions
        questions = [q.strip() for q in questions if q.strip()]  # Clean up empty lines
        return questions[:num_questions]  # Return only the desired number of questions
    except Exception as e:
        logging.error(f"Error generating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

# Parse PDF resume and fallback to OCR if no text is found
def parse_resume(pdf_file: UploadFile):
    try:
        logging.info(f"Attempting to parse file: {pdf_file.filename}")
        if pdf_file.filename.endswith(".pdf"):
            # Try to extract text using PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file.file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            if not text:
                logging.info("No text found using PyPDF2, trying OCR.")
                text = ocr_extract_text_from_pdf(pdf_file)  # Fallback to OCR
            if not text:
                raise ValueError("No text found in the PDF.")
            logging.info(f"Parsed text length: {len(text)} characters")
            return text
        else:
            raise ValueError(f"Unsupported file type: {pdf_file.filename}")
    except Exception as e:
        logging.error(f"Error parsing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

# OCR function to extract text from PDF using pytesseract
def ocr_extract_text_from_pdf(pdf_file: UploadFile):
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_file.file.name)
        text = ""
        for image in images:
            # Perform OCR on each page
            page_text = pytesseract.image_to_string(image)
            text += page_text
        return text
    except Exception as e:
        logging.error(f"Error during OCR extraction: {str(e)}")
        return ""

# Store the conversation flow locally
def store_conversation(conversation_data: dict):
    try:
        with open("conversation_log.json", "w") as f:
            json.dump(conversation_data, f, indent=4)
        logging.info("Conversation stored successfully.")
    except Exception as e:
        logging.error(f"Error storing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store conversation.")

# API endpoint to handle resume upload and question generation
@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...), num_questions: int = 5):
    # Parse the resume
    resume_text = parse_resume(file)
    
    # Generate questions
    questions = generate_questions_tool(resume_text, num_questions)

    return {"questions": questions, "resume_text": resume_text}

# WebSocket endpoint to handle interview session (question/answer)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients_data[websocket] = {"questions": [], "answers": [], "current_question_index": 0}

    try:
        while True:
            data = await websocket.receive_text()

            if data.startswith("START_INTERVIEW"):
                resume_text = clients_data[websocket]["resume_text"]
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
