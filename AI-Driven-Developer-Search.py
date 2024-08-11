import streamlit as st
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import spacy
import os  # Import the os module to use getenv for environment variables
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from transformers import pipeline
import json
load_dotenv()  # Load environment variables from the .env file

# Initialize the ChatOpenAI model with your API key
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openai_api_key, temperature=0.5, model_name="gpt-4")

# MongoDB setup
connection_string = os.getenv('MONGO_CONNECTION_STRING')
client = MongoClient(connection_string)
db = client['user_management']
collection = db['engineerbasicinfos']

# Set up a simple NER tool using Hugging Face's pipeline
ner_pipeline = pipeline("ner", grouped_entities=True)

# Define a simple prompt template for extracting entities
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Extract the entities like skills, experience, and location from the following text: {text}"
)

# Combine the prompt template with the LLM
llm_chain = LLMChain(llm=llm, prompt=prompt_template)
ner_pipeline = pipeline("ner", grouped_entities=True)


def extract_entities(query):
    # Combine the prompt template with the LLM
    llm_result = llm_chain.run({"text": query})
    
    # Print and write the LLM result to debug
    print('llm_result:', llm_result)
    st.write("---", llm_result)
    
    try:
        # Extract the required information using regular expressions
        skills = re.search(r'Skills:\s*(.*)', llm_result)
        experience = re.search(r'Experience:\s*(.*)', llm_result)
        location = re.search(r'Location:\s*(.*)', llm_result)
        
        # Process the experience to remove the word "years" and convert to a numeric value
        if experience:
            experience_value_str = re.sub(r'\s*years?\s*', '', experience.group(1))
            try:
                experience_value = int(experience_value_str)  # Convert to an integer
            except ValueError:
                experience_value = float(experience_value_str)  # Convert to a float if it's not an integer
        else:
            experience_value = None  # Use None if experience is not found or cannot be converted

        # Get the matched groups or return empty strings if not found
        return (
            skills.group(1) if skills else "",
            experience_value,
            location.group(1) if location else ""
        )
        
    except Exception as e:
        print(f"Error: {e}")
        
        # Return empty strings in case of any other errors
        return ("", None, "")
        



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

def mongo_search_dynamic(query, skills, experience, location):
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

    # if location:
    #     pipeline.append({
    #         "$match": {
    #             "location": {"$regex": location, "$options": "i"}
    #         }
    #     })

    pipeline.append({
        "$project": {
            "first_name": 1,
            "last_name": 1,
            "primary_skills": 1,
            "work_experience": 1,
            "location": 1,
            "job_title": 1
        }
    })


    print(f"MongoDB query pipeline: {pipeline}")
    return list(collection.aggregate(pipeline))





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
"You are an assistant that checks for three key elements: Location, Experience, and Skills. If any of these are not present, ask a question to gather the missing information. Answer the following question: {query}"        )
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

def perform_search_with_clarifications(query, clarification):
    """Perform a search with clarified requirements."""
    if clarification:
        query += " " + clarification
    search_results = mongo_search_dynamic(query)
    return search_results

# Streamlit App
st.title("Developer Search Interface")

user_query = st.text_input("Enter your search query:")

if st.button("Search"):
    if user_query:
        skills, experience, location = extract_entities(user_query)
        print('skills, experience, location',skills, experience, location)
        search_results = mongo_search_dynamic(user_query,skills, experience, location)
        print('search_results',search_results)
        if skills and experience or skills and experience and location: 
            documents = [
                f"{profile.get('first_name', '')} {profile.get('last_name', '')}, "
                f"Skills: {', '.join(str(skill.get('skill_name', 'Not specified') or 'Not specified') for skill in profile.get('primary_skills', []))}, "
                f"Experience: {profile.get('work_experience', 'Not specified')} years, "
                f"Location: {profile.get('location', 'Not specified')}, "
                f"Job Title: {profile.get('job_title', 'Not specified')}"
                for profile in search_results
            ]
            if len(search_results):
                cosine_sim = calculate_cosine_similarity(user_query, documents)
                top_indices = cosine_sim.argsort()[-5:][::-1]
                top_profiles = [search_results[i] for i in top_indices]
                st.write("Top 5 Matching Candidates:")
                for i, profile in enumerate(top_profiles):
                    st.write(f"Name: {profile.get('first_name', 'Not specified')} {profile.get('last_name', 'Not specified')}")
                    st.write(f"Skills: {', '.join(str(skill.get('skill_name', 'Not specified') or 'Not specified') for skill in profile.get('primary_skills', []))}")
                    st.write(f"Experience: {profile.get('work_experience', 'Not specified')} years")
                    st.write(f"Location: {profile.get('location', 'Not specified')}")
                    st.write(f"Job Title: {profile.get('job_title', 'Not specified')}")
                    st.write(f"Cosine Similarity Score: {cosine_sim[i].round(2)}")
                    st.write("---")
            else:
                st.write("No candidates found for your query.")

        else:
            clarification = ask_clarifying_questions(user_query)
            st.write(f"Clarification needed: {clarification}")
            follow_up_query = st.text_input("Please provide more details:")
            if follow_up_query:
                user_query += " " + follow_up_query
                search_results = perform_search_with_clarifications(user_query, clarification)
                if search_results:
                    documents = [
                        f"{profile.get('first_name', '')} {profile.get('last_name', '')}, "
                        f"Skills: {', '.join(str(skill.get('skill_name', 'Not specified') or 'Not specified') for skill in profile.get('primary_skills', []))}, "
                        f"Experience: {profile.get('work_experience', 'Not specified')} years, "
                        f"Location: {profile.get('location', 'Not specified')}, "
                        f"Job Title: {profile.get('job_title', 'Not specified')}"
                        for profile in search_results
                    ]
                    
                    cosine_sim = calculate_cosine_similarity(user_query, documents)
                    top_indices = cosine_sim.argsort()[-5:][::-1]
                    top_profiles = [search_results[i] for i in top_indices]
                    st.write("Top 5 Matching Candidates:")
                    for i, profile in enumerate(top_profiles):
                        st.write(f"Name: {profile.get('first_name', 'Not specified')} {profile.get('last_name', 'Not specified')}")
                        st.write(f"Skills: {', '.join(str(skill.get('skill_name', 'Not specified') or 'Not specified') for skill in profile.get('primary_skills', []))}")
                        st.write(f"Experience: {profile.get('work_experience', 'Not specified')} years")
                        st.write(f"Location: {profile.get('location', 'Not specified')}")
                        st.write(f"Job Title: {profile.get('job_title', 'Not specified')}")
                        st.write(f"Cosine Similarity Score: {cosine_sim[i].round(2)}")
                        st.write("---")
                else:
                    st.write("No candidates found for your query.")
    else:
        st.write("Please enter a search query.")
