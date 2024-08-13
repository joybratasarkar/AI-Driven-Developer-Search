# AI-Driven Developer Search

This project is an AI-powered developer search tool built with Streamlit, MongoDB, and OpenAI's GPT-4 model. The application allows users to search for developers based on specific skills, experience, and location criteria. The tool leverages natural language processing (NLP) techniques to extract relevant information from user queries and provides the top matching candidates from a MongoDB database.

## Features

- **Natural Language Querying**: Users can enter search queries in natural language, and the application will extract key information like skills, experience, and location.
- **Cosine Similarity Matching**: The tool uses TF-IDF vectorization and cosine similarity to rank and retrieve the top 5 matching candidates based on the query.
- **Dynamic Query Refinement**: If the initial search query is incomplete, the system can ask clarifying questions to gather the necessary information.
- **Interactive Interface**: The application is built with Streamlit, offering a user-friendly interface for interacting with the search tool.

## Installation

### Prerequisites

- Python 3.8+
- MongoDB
- Git
- Virtual environment tool (optional but recommended)

### Clone the Repository

# AI-Powered Resume Search

This project is an AI-powered resume search tool that allows you to efficiently search through a database of resumes based on natural language queries. The application uses Sentence Transformers for embedding resumes and FAISS for fast similarity search, making it possible to find the most relevant resumes for a given job description or query.

## Features

- **Resume Embedding**: Converts resumes into dense vector representations using Sentence Transformers.
- **FAISS Indexing**: Uses FAISS (Facebook AI Similarity Search) to create a vector index for fast similarity search.
- **Custom Query Search**: Allows users to input a natural language query, and the system will retrieve the top matching resumes based on cosine similarity.
- **GPU Acceleration**: Supports GPU for faster processing if available, falling back to CPU otherwise.

## Installation

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool (optional but recommended)
- A CSV file named `Resume.csv` containing resume data with columns `ID`, `Resume_str`, and `Category`.

### Clone the Repository

```bash
git clone https://github.com/joybratasarkar/AI-Powered-Resume-Search.git
cd AI-Powered-Resume-Search



## How to Use This `README.md`

1. Save the provided text into a file named `README.md` in the root directory of your project.
2. If your project involves environment variables (such as OpenAI API keys or MongoDB connection strings), make sure to replace any placeholder text like `your_openai_api_key` and `your_mongo_connection_string` with the actual values in your `.env` file.
3. Ensure that any necessary data files (like `Resume.csv`) are in the correct format as described in the README.
4. Once you've made these updates, commit the `README.md` file to your repository and push it to GitHub.

This README should provide a comprehensive guide for anyone who wants to understand, set up, and use your project, whether it's for AI-Powered Developer Search or AI-Powered Resume Search.



