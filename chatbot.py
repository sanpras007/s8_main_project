import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")

# Set up Google AI API key
genai.configure(api_key=key)

# Choose an available model
MODEL_NAME = "models/gemini-1.5-pro-latest"  # Replace with any available model

# Load the model
model = genai.GenerativeModel(MODEL_NAME)

# Example function to get feedback
def get_answer_feedback(correct_answer, student_answer, score, max_score):
    prompt = f"""
    You are an AI evaluator for SmartScribe, a subjective answer evaluation system.
    
    Task:
    - Analyze the student's answer based on accuracy, clarity, and completeness.
    - Compare it with the correct answer.
    - Provide structured feedback in a clear and readable format.
    
    Inputs:
    Correct Answer: {correct_answer}
    Student's Answer: {student_answer}
    maximum mark for the question : {max_score}
    Score assigned by BERT & SBERT combined: {score}
    
    Expected Output:
    1. Reason for marks deduction (if any).
    2. Suggestions for improvement.
    3. If the answer is well-written and have a good score, give positive reinforcement.

    Format the response in simple text,
    without symbols or markdown,
    separate sections using blank lines,
    keep it simple and informative in min words.
    """
    response = model.generate_content(prompt)
    return response.text  # Extract generated text

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

    except Exception as e:
        print(f"❌ Error generating AI response: {e}")
        return "Error generating AI response"

