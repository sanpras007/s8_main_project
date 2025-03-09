from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
import asyncio
import bcrypt
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB
async def connect_db():
    uri = os.getenv("MONGO_URI")
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
        
        if isinstance(data, list): 
            if data:  # Ensure data is not empty
                result = collection.insert_many(data)
                print(f"Inserted {len(result.inserted_ids)} documents.")
            else:
                print("⚠️ No valid data to insert.")
        elif isinstance(data, dict): 
            result = collection.insert_one(data)
            print(f"Inserted 1 document with ID: {result.inserted_id}")
        else:
            raise ValueError("Data must be a dictionary or list of dictionaries.")
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




#-------------------------|
# USER SIGN IN AND LOGIN  |
#-------------------------|


async def create_user(name, email, password):
    """Hash password and store user in DB."""
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        db = await connect_db()
        users_collection = db["users"]
        user = {
            "name": name,
            "email": email,
            "password": hashed_pw
        }
        
        if users_collection.find_one({"email": email}):
            return {"error": "User already exists"}
        
        users_collection.insert_one(user)
        return {"message": "User registered successfully"}
    except Exception as e:
        print(f"⚠️ Error creating data: {e}")

async def get_user_by_email(email):
    """Retrieve a user by email (excluding `_id`)."""
    try:
        db = await connect_db()
        users_collection = db["users"]
        return users_collection.find_one({"email": email}, {"_id": 0})  # Exclude `_id`
    except Exception as e:
        print(f"⚠️ Error creating data: {e}")