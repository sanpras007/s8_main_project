import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
import time

load_dotenv()

key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")

# Set up Google AI API key
genai.configure(api_key=key)

# Choose an available model
MODEL_NAME = "models/gemini-1.5-flash"  # Replace with any available model

# Load the model
model = genai.GenerativeModel(MODEL_NAME)

# Example function to get feedback
def get_answer_feedback(correct_answer, student_answer, score, max_score):
    try:
        prompt = prompt = f"""
            Evaluate student answer.
            Correct: {correct_answer}
            Student: {student_answer}
            Max Score: {max_score}
            Score: {score}
            Give feedback in 3 points: deduction reason, improvements, praise (if applicable) dont need decorations.
            """
        response = model.generate_content(prompt)
        return response.text  # Extract generated text
    except ResourceExhausted:
        print("Quota exceeded! Retrying after delay...")
        time.sleep(10)  # Wait before retrying
        return "Quota exceeded. Please try again later."

def chat(user_message):
    try:
        prompt = f"""
        You are SmartScribe, an AI-powered assistant designed to help users with SmartScribe, a subjective answer evaluation system. 

        Your role is to provide clear, concise, and helpful responses about the app’s functionalities, including:
        - Answer evaluation using OCR and AI models (BERT & SBERT).
        - Generating feedback for students to improve their responses.
        - Guiding users on how to upload answer sheets and use the platform.
        - Explaining SmartScribe’s grading methodology.
        - Addressing common issues like incorrect OCR results or evaluation errors.
        - Assisting with integration and API-related queries.

        Follow these response guidelines:
        1. Be professional, user-friendly, and informative.
        2. keep the reply very short and structured 
        3. If a question is unclear, ask for clarification.
        4. Provide step-by-step guidance when necessary.
        5. reply as a chatbot in sentances only
        
        Question: {user_message}
        """
        response = model.generate_content(prompt)
        return response.text
    except ResourceExhausted:
        print("Quota exceeded! Retrying after delay...")
        time.sleep(10)  # Wait before retrying
        return "Quota exceeded. Please try again later."
    except Exception as e:
        print(f"❌ Error generating AI response: {e}")
        return "Error generating AI response"

