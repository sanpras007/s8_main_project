from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

# Connect to MongoDB
async def connect_db():
    uri = "mongodb+srv://21cs445:KVgPv5yxySNKaH9m@cluster0.uz0ck.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri, server_api=ServerApi('1'))
    return client["grading_system"]

async def retrieve_all_answers(db_name):
    """Retrieve all questions, answers, and max marks from the database."""
    try:
        db = await connect_db()
        collection = db[db_name]

        results = collection.find({}, {"_id": 0})  # Exclude `_id` field

        all_answers = {}
        for result in results:
            question_number = result.get("question_number")
            if question_number:
                all_answers[question_number] = {
                    "model_answer": result.get("model_answer", ""),
                    "max_marks": result.get("max_marks", 0)
                }
        
        return all_answers
    except Exception as e:
        print(f"⚠️ Error retrieving all data: {e}")
        return {}

async def retrieve_answer_and_marks(db_name, question_number):
    try:
        db = await connect_db()
        collection = db[db_name]
        
        result = collection.find_one({"question_number": question_number}, {"_id": 0})  # Exclude `_id` field
        
        if result:
            return result.get("model_answer"), result.get("max_marks")
        return None, None  # If no document found, return None
    except Exception as e:
        print(f"⚠️ Error retrieving data: {e}")
        return None, None

async def insert_to_db(db_name, data):
    try:
        db = await connect_db()
        collection = db[db_name]
        
        if isinstance(data, list):  # Check if data is a list for bulk insert
            collection.insert_many(data)
        elif isinstance(data, dict):  # Handle single document insert
            collection.insert_one(data)
        else:
            raise ValueError("Data must be a dictionary or list of dictionaries.")
        
        print(f"✅ Successfully added data to database: {db_name}.")
    except Exception as e:
        print(f"⚠️ Error inserting data: {e}")



async def delete_all(db_name):
    try:
        db = await connect_db()
        collection = db[db_name]
        result = collection.delete_many({})
        print(f"✅ Deleted {result.deleted_count} documents from '{db_name}' collection.")
    except Exception as e:
        print(f"⚠️ Error deleting data: {e}")


