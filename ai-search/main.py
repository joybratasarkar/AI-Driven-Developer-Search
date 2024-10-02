from fastapi import FastAPI, HTTPException, Query
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import spacy
import os
from dotenv import load_dotenv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import logging
from pydantic import BaseModel
from typing import List, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from schema import ReportSchema, report_collection
app = FastAPI()

# Enable CORS if your frontend is served on a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the ChatOpenAI model with your API key
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openai_api_key, temperature=0.5, model_name="gpt-4")

# MongoDB setup
connection_string = os.getenv('MONGO_CONNECTION_STRING')
client = MongoClient(connection_string)
db = client['user_management']
collection = db['engineerbasicinfos']

# Function to count the number of candidates in the database
def count_candidates():
    count = collection.count_documents({})
    logging.info(f"Number of candidates in the database: {count}")
    return count

# FastAPI startup event to print the number of candidates
@app.on_event("startup")
async def startup_event():
    count = count_candidates()
    print(f"Number of candidates in the database: {count}")

# Set up a simple NER tool using Hugging Face's pipeline
ner_pipeline = pipeline("ner", grouped_entities=True)

# Define a simple prompt template for extracting entities
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Extract the entities like skills, experience, and location from the following text: {text}"
)

# Combine the prompt template with the LLM
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

class SearchRequest(BaseModel):
    query: str
    follow_up: Optional[str] = None

@app.post("/search")
def search_candidates(request: SearchRequest):
    print('-----------')
    query = request.query
    follow_up = request.follow_up
    
    if follow_up:
        query += " " + follow_up
    
    skills, experience, location = extract_entities(query)
    logging.info(f'Skills: {skills}, Experience: {experience}, Location: {location}')
    
    search_results = mongo_search_dynamic(query, skills, experience, location)
    # print('search_results=======================================================',search_results)
    # logging.info(f'Search results: {search_results}')
    
    if search_results:
        documents = [
    (
        f"{profile.get('first_name', '')} {profile.get('last_name', '')}, "
        f"Skills: {', '.join(skill.get('skill_name', 'Not specified') for skill in profile.get('primary_skills', []))}, "
        f"Experience: {profile.get('work_experience', 'Not specified')} years, "
        f"Location: {profile.get('location', 'Not specified')}, "
        f"engineer_id: {str(profile.get('_id', 'Not specified'))}, "
    )
    for profile in search_results
]

        # print('--------------------------------------------------------------------',documents)
        
        cosine_sim = calculate_cosine_similarity(query, documents)
        top_indices = cosine_sim.argsort()[-5:][::-1]
        top_profiles = [search_results[i] for i in top_indices]
        # print('-------------------------------',search_results)
        return {
            "top_profiles": [
            {
                "name": f"{profile.get('first_name', 'Not specified')} {profile.get('last_name', 'Not specified')}",
                "skills": [skill.get('skill_name', 'Not specified') for skill in profile.get('primary_skills', [])],
                "experience": profile.get('work_experience', 'Not specified'),
                "location": profile.get('location', 'Not specified'),
                "engineer_id": str(profile.get('_id', 'Not specified')),  # Corrected to fetch the MongoDB ObjectId
                # "job_title": profile.get('job_title', 'Not specified'),  # Uncomment if needed and ensure correct path
                "cosine_similarity_score": cosine_sim[i].round(2)
            }
    for i, profile in enumerate(top_profiles)
]

        }
    else:
        clarification = ask_clarifying_questions(query)
        return {"clarification_needed": clarification}

def extract_entities(query):
    """Extract entities such as skills, experience, and location from the query."""
    try:
        llm_result = llm_chain.run({"text": query})
        logging.info(f'LLM result: {llm_result}')
        
        skills = re.search(r'Skills:\s*(.*)', llm_result)
        experience = re.search(r'Experience:\s*(.*)', llm_result)
        location = re.search(r'Location:\s*(.*)', llm_result)
        
        experience_value = None
        if experience:
            experience_value_str = re.sub(r'\s*years?\s*', '', experience.group(1))
            # Validate if the experience string is numeric
            if experience_value_str.isdigit():
                experience_value = int(experience_value_str)
            else:
                try:
                    # Attempt to convert to float if it is not an integer
                    experience_value = float(experience_value_str)
                except ValueError:
                    experience_value = None
        
        return (
            skills.group(1) if skills else "",
            experience_value,
            location.group(1) if location else ""
        )
    except Exception as e:
        logging.error(f"Error extracting entities: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while extracting entities: {e}")

def build_experience_filter(experience, comparison="exactly"):
    """Build the experience filter for MongoDB."""
    if experience is None:
        return {}
    if comparison == "less than":
        return {"$lt": experience + 1}
    elif comparison == "more than":
        return {"$gte": experience}
    elif comparison == "exactly":
        return {"$eq": experience}
    else:
        return {"$eq": experience}  # Default to exact match

def mongo_search_dynamic(query, skills="", experience=None, location=""):
    """Perform a MongoDB search with dynamic filtering based on skills, experience, and location."""
    experience_filter = build_experience_filter(experience)
    pipeline = [
        {
            "$match": {
                "$text": {
                    "$search": query
                }
            }
        }
    ]

    if experience_filter:
        pipeline.append({
            "$match": {
                "work_experience": experience_filter
            }
        })

    pipeline.append({
        "$project": {
            "first_name": 1,
            "last_name": 1,
            "primary_skills": 1,
            "work_experience": 1,
            "location": 1,
            "_id": 1,
            "project_id": 1
        }
    })
    
    logging.info(f"MongoDB query pipeline: {pipeline}")
    result = list(collection.aggregate(pipeline))
    
    # Debugging: Check if the result contains the expected fields
    for doc in result:
        if 'engineer_id' not in doc or 'project_id' not in doc:
            logging.warning(f"Document missing expected fields: {doc}")

    return result


def calculate_cosine_similarity(query, documents):
    """Calculate cosine similarity between the query and each document."""
    vectorizer = TfidfVectorizer().fit_transform([query] + documents)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return cosine_sim

class QnABot:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=openai_api_key, temperature=0.5, model_name="gpt-4")
        self.prompt_template = PromptTemplate.from_template(
            "You are an assistant that checks for three key elements: Location, Experience, and Skills. If any of these are not present, ask a question to gather the missing information. Answer the following question: {query}"
        )
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )

    def answer_query(self, query):
        """Process the query through the Q&A chain."""
        response = self.chain.run({"query": query})
        return response

    def get_candidate_requirements(self, query):
        """Generate clarifying questions based on the user query."""
        response = self.answer_query(query)
        return response

def ask_clarifying_questions(query):
    """Ask clarifying questions based on the user query using LLM."""
    qna_bot = QnABot()
    clarifying_question = qna_bot.get_candidate_requirements(query)
    return clarifying_question


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