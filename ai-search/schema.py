from pydantic import BaseModel, Field
from typing import Optional
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

# Define the schema for the report
class ReportSchema(BaseModel):
    results: str = Field(..., description="The summary of the search results as a list of dictionaries.")
    engineer_id: object = Field(..., description="engineer id")

# MongoDB connection setup
mongo_uri = os.getenv("MONGODB_URI_local")
client = MongoClient(mongo_uri)
db = client.get_database('ai-interview-questions')  # Use the correct database name
report_collection = db.get_collection("reports")  # Collection to save reports
