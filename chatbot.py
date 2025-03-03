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
def get_answer_feedback(correct_answer, student_answer, score):
    prompt = f"""
    Evaluate the student's answer and provide feedback based on accuracy, clarity, and completeness.
    
    **Correct Answer:** {correct_answer}
    **Student's Answer:** {student_answer}
    **score given my model BERT and SBERT combined:** {score}
    
    1. Explain why marks were lost (if any).  
    2. Suggest how to improve the answer.  
    3. If the answer is good, give positive feedback.  

    Keep the feedback simple and to the point no many words needed just the points dont want it as bold just as sentance.
    """
    response = model.generate_content(prompt)
    return response.text  # Extract generated text

