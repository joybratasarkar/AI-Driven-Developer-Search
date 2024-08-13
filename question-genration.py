from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import openai
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Hugging Face summarization pipeline with an appropriate model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize OpenAI LLM
llm = OpenAI()

# Define the prompt template for generating interview questions
question_prompt_template = """
You are a professional interviewer tasked with creating interview questions for a job role. Based on the summarized job description provided below, generate a list of interview questions categorized into Easy, Medium, and Hard. Each category should include relevant and insightful questions that help assess the candidate's skills and fit for the role.

Job Description Summary:
{job_description_summary}

Questions:
- Easy:
- Medium:
- Hard:
"""

# Create a PromptTemplate instance for generating interview questions
question_prompt = PromptTemplate(template=question_prompt_template, input_variables=["job_description_summary"])

def summarize_job_description(job_description):
    # Determine the length of the job description
    input_length = len(job_description.split())
    
    # Set max_length and min_length based on input_length
    if input_length > 500:
        max_length = 150
        min_length = 50
    elif input_length > 300:
        max_length = 100
        min_length = 30
    else:
        max_length = 80
        min_length = 20

    # Use the Hugging Face summarization model to summarize the job description
    summary = summarizer(job_description, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def generate_interview_questions(job_description_summary):
    # Generate interview questions based on the summarized job description using the prompt template and LLM
    prompt = question_prompt.format(job_description_summary=job_description_summary)
    questions = llm(prompt)
    return questions.strip()

def main():
    # Get user input for job description
    job_description = """
    The rapid growth of the Microsoft Power Platform (Power Apps, Power Pages, Power Automate, Power Virtual Agent, Dataverse, Power BI) is fueled by organizations investing in leveraging low-code/no-code development models to accelerate digital transformation. Power Platform is a strategic growth for Microsoft, giving users access to create next-generation business productivity software via simplified experiences with the help of AI Copilot, without requiring extensive training. It transforms the careers of 'citizen developers,' enhances 'pro-developers' productivity, and helps IT innovate at the pace businesses expect. Microsoft is a leader in SaaS business applications, and the Power Platform is central to designing and delivering business applications, making it one of the fastest-growing products at Microsoft and in the industry.

    Are you a great Frontend Software Engineer interested in working with a dynamic technical team and passionate about the latest technologies? Join the Dataverse Experiences Team! This team builds high-quality user experiences across the Dataverse ecosystem, using modern Power Apps Component Framework (PCF) technology, migrating legacy experiences, and implementing scalable and performant solutions. You'll collaborate in a startup-like environment with global teams, including Power Platform, Azure, Microsoft Teams, Office, and others. Our culture promotes diversity, multiculturalism, and inclusion. We value accountability, learning opportunities, and career growth.

    Qualifications:
    - In-depth professional software development experience.
    - Strong grasp of JavaScript, TypeScript, CSS.
    - Experience with modern client-side/web technologies such as React, JavaScript, TypeScript, Webpack, Node.js, or similar.
    - Strong object-oriented programming skills.
    - Strong software design, analytical, and problem-solving skills.
    - Ability to meet tight deadlines, follow development standards, and work independently.
    - Business-level English proficiency.
    - Passionate about solving challenging problems.
    - BS/MS in Computer Science or equivalent.
    - Experience in Agile and DevOps methodologies.

    Preferred Qualifications:
    - Strong experience with C#.
    - Experience with Dynamics 365, Model-driven Power Apps, Power Apps Component Framework (PCF).
    - Experience with Azure cloud technologies, SQL, NoSQL, Microservices, ML, or AI.

    Responsibilities:
    - Build innovative user experiences.
    - Lead technical direction, architecture, design, and implementation of Dataverse Experiences features.
    - Ensure uninterrupted operation of features in the production environment.
    - Collaborate with internal partner teams like Power Platform, Azure, Microsoft Teams, Office, and others.
    - Interact with customers and partners to empower and learn from them.
    - Mentor junior engineers for growth.
    """

    # Summarize the job description
    job_description_summary = summarize_job_description(job_description)
    print("\nJob Description Summary:")
    print(job_description_summary)
    print("\nGenerating interview questions...\n")

    # Generate interview questions based on the summary
    questions = generate_interview_questions(job_description_summary)
    print("Interview Questions:")
    print(questions)

if __name__ == "__main__":
    main()
