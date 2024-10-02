import requests
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
import os
from multiprocessing import Pool, cpu_count

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
        summary = llm.invoke(f"Refine the following summary to make it more concise and clear in brief :\n\n{summary.content}")
    
    return summary.content

# Function to use LLM to evaluate the similarity, passing role and key factors
def evaluate_answer_with_llm(candidate_answer, correct_answer, role, key_factors):
    prompt = (
        f"You are an AI assistant helping recruiters evaluate a candidate's fit for the role of {role}. "
        f"Consider the following key factors while evaluating the candidate's answer:\n\n"
        f"{key_factors}\n\n"
        
        f"1. **Technical Skills**\n"
        f"   - Relevance to Role: How well does the candidate's answer demonstrate skills that are directly relevant to the job requirements?\n"
        f"   - Depth of Knowledge: Does the candidate's answer reflect a deep understanding and expertise in the subject matter?\n"
        f"   - Problem-Solving Ability: How effectively does the candidate approach and solve the problem presented in the question?\n\n"
        
        f"2. **Experience**\n"
        f"   - Relevant Work Experience: How well does the answer showcase experience that is applicable to the job?\n"
        f"   - Adaptability and Learning: Does the answer indicate that the candidate can adapt to new challenges and learn new skills?\n\n"
        
        f"3. **Soft Skills**\n"
        f"   - Communication Skills: Is the answer clear, well-structured, and easy to understand?\n"
        f"   - Teamwork and Collaboration: Does the answer suggest the candidate's ability to work effectively within a team?\n"
        f"   - Cultural Fit: How well does the candidate's answer align with the company's values and culture?\n\n"
        
        f"4. **Behavioral Competencies**\n"
        f"   - Leadership and Initiative: Does the answer demonstrate leadership qualities or the ability to take initiative?\n"
        f"   - Work Ethic and Attitude: What does the answer reveal about the candidate's work ethic and attitude towards their work?\n"
        f"   - Decision-Making Skills: How well does the candidate make decisions or justify their approach in the answer?\n\n"
        
        f"Candidate's Answer: {candidate_answer}\n\n"
        f"Correct Answer: {correct_answer}\n\n"
        
        f"Provide a detailed textual evaluation of the candidate's strengths and weaknesses in relation to the role of {role}, focusing on the key factors provided and other aspects mentioned above. "
        f"Do not provide a score."
    )

    response = llm.invoke(prompt)
    return response.content


# Function to process each question-answer pair
def process_question_answer(item):
    question = item['question']
    candidate_answer = item['answer']

    try:
        # Retrieve the correct answer dynamically using the agent
        result = agent.invoke(f"Search the internet to answer: {question}")
        correct_answer = result['output']
        print(f"Correct answer retrieved from the internet for question '{question}': {correct_answer}")

        # Summarize the correct answer using Refine Approach
        summarized_correct_answer = refine_summary(correct_answer)
        print(f"Summarized correct answer: {summarized_correct_answer}")

        # Use LLM to evaluate the candidate's answer
        role = "Senior Frontend Developer"
        key_factors = "The candidate should demonstrate experience with modern frontend technologies such as React, Angular, or Vue, as well as strong proficiency in JavaScript, CSS, and responsive design."

        evaluation_text = evaluate_answer_with_llm(candidate_answer, summarized_correct_answer, role, key_factors)
        print(f"Textual evaluation for question '{question}': {evaluation_text}")

        return (question, candidate_answer, evaluation_text)

    except Exception as e:
        print(f"An error occurred for question '{question}': {e}")
        return (question, candidate_answer, "Error")


# Function to generate recruiter report
def generate_recruiter_report(results, job_description):
    evaluations = []
    
    for question, candidate_answer, evaluation_text in results:
        evaluations.append((question, candidate_answer, evaluation_text))

    # Prompt for generating a final summary tailored for a recruiter
    report_prompt = (
    "Based on the following candidate evaluations and job description, generate a detailed recruiter report with ratings (1 to 5) for each section and explain the reasoning:\n\n"
    f"Job Description: {job_description}\n\n"
    + "\n".join([f"Question: {q}\nAnswer: {a}\nEvaluation: {e}" for q, a, e in evaluations]) +
    "\n\nGenerate the report in the following format:\n\n"
    "---\n"
    "Recruiter Report**\n\n"
    "Each section must have a minimum of 2 points and a maximum of 5 points.\n\n"
    
    "Section 1:\n"
    "Cultural Alignment (1-5): Rating and reasoning.\n"
    "Motivating Factors: Key motivating factors based on the candidate's responses.\n"
    "Suitable Roles:** Suggested roles based on the candidate's experience and skills.\n\n"
    
    "Section 2:\n"
    "- Technical PROS & CONS (1-5):\n"
    "  PROS:* Mention 2 to 5 technical strengths.\n"
    "  CONS:* Highlight 2 to 5 areas where improvement is needed.\n"
    "- Communication PROS & CONS (1-5):\n"
    "  PROS:* Summarize 2 to 5 communication strengths.\n"
    "  CONS:* Note 2 to 5 areas for improvement.\n"
    "- *Trustworthiness PROS & CONS (1-5):\n"
    "  PROS:* List 2 to 5 positive traits related to trustworthiness.\n"
    "  CONS:* Mention 2 to 5 concerns if any.\n"
    "---"
)


    report_response = llm.invoke(report_prompt)
    return report_response.content


# Sample conversation history data
conversation_history = [
    {'question': 'Tell me about a time when you had to quickly learn a new technology. How did you handle it?', 
     'answer': 'In my previous role, we needed to implement a microservices architecture for a new project, and I had little prior experience with it. I started by researching and studying the fundamentals, reading documentation, and following tutorials. I also set up a small prototype to get hands-on experience. Within two weeks, I was able to implement a microservices solution that integrated seamlessly with our existing infrastructure. The project was delivered on time, and I received positive feedback from both my team and stakeholders.'},
    
    {'question': 'How do you prioritize tasks when you are working on multiple projects with tight deadlines?', 
     'answer': 'I prioritize tasks based on their urgency and impact. I start by breaking down each project into smaller tasks and then identify which tasks are time-sensitive and critical for moving the project forward. I use project management tools like Trello to organize and track my progress. I also make sure to communicate with stakeholders regularly to align on priorities and timelines. If needed, I’m not afraid to ask for help or delegate tasks to ensure everything gets completed on time.'}
]





# Sample Job Description
job_description = """
As a Senior Frontend Software Engineer on the Microsoft team, you will:    

Work with engineers, program managers, designers, and partners to deliver sets of features with the right overall design and architecture. 
Own and deliver complete feature areas and large-scale solutions including design, architecture, implementation, testability, debugging, and shipping with secure long-term support. 
As a technical leader on the team, you will be responsible for sharing insights and best practices that can be applied to improve development and influence direction across related sets of systems, platforms, and/or products. 
Continue to develop your approaches through interactions with more experienced team members and continually reviewing processes to ensure efficiency. 
Provide mentorship and coaching to more junior engineers to help them identify and propose relevant solutions. 
Write clean well-thought-out code with an emphasis on quality, performance, simplicity, durability, scalability, reusability, and maintainability. 
Be committed to delivering the best experience for our customers and partners, and then iterate based on qualitative and quantitative feedback. 
Help create a diverse and inclusive culture, participating in hiring where appropriate, so everyone can bring their full and authentic self and where we do our best work as a result. 
Take responsibility for reliable uninterrupted operation of features with the earliest detection of issues in production.  
Foster a data driven approach with everything you do from analysing and prioritizing business requirements, including customer feedback, and using metrics to prove success. 
Expose test coverage issues, organize, and implement tests and types of tests needed, and resolve problem areas. 
Gain a working understanding of Microsoft businesses and collaborate with mentors and leaders to contribute to cohesive, end-to-end experiences for our customers and users. 
"""

if __name__ == "__main__":
    # Utilize multiprocessing to process the conversation history faster
    with Pool(cpu_count()) as pool:
        results = pool.map(process_question_answer, conversation_history)

    # Generate and print the recruiter report
    final_report = generate_recruiter_report(results, job_description)
    print(f"Final Recruiter Report:\n{final_report}")
