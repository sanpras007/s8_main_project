import tensorflow as tf
import transformers
import asyncio
from flask import Flask,flash, render_template, request
from DB_operations import insert_to_db, delete_all, retrieve_all_answers,create_user, get_user_by_email
from database import client
import numpy as np
from huggingface_hub import from_pretrained_keras
import tensorflow_hub as hub
from transformers import AutoModel, AutoTokenizer
import os as os
import bcrypt
import re
import torch
import fitz
from motor.motor_asyncio import AsyncIOMotorClient
from flask_cors import CORS  # Import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from chatbot import get_answer_feedback,chat
from flask import Blueprint, request, jsonify
from pdf2image import convert_from_path
from PIL import Image



app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "http://127.0.0.1:5000"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



bert_model = from_pretrained_keras("keras-io/bert-semantic-similarity")
labels = ["Contradiction", "Perfect", "Neutral"]

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
ocr_model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
ocr_model = ocr_model.eval().to(device)


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data."""

    def __init__(
            self,
            sentence_pairs,
            labels,
            batch_size=32,
            shuffle=True,
            include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=128,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]


@app.route("/")
def home():
    return render_template("main.html")


@app.route("/checkMyAnswer")
def checkMyAnswer():
    return render_template("checkMyAnswer.html")

@app.route("/answers")
def answers():
    return render_template("answers.html")

@app.route("/upload")
def upolad_check():
    return render_template("uplod_check.html")

@app.route("/answerkey_upload")
def answerkey_upload():
    return render_template("ans_key_upload_check.html")



def grade_answer(answer_key, student_ans):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    key_embedding = model.encode([answer_key])
    student_embedding = model.encode([student_ans])
    similarity = cosine_similarity(key_embedding, student_embedding)[0][0]
    return round(similarity * 100, 2)  # Convert to percentage

def final_score(answer_key, student_ans):
    sbert_score = grade_answer(answer_key, student_ans)
    match_score = check_similarity(answer_key, student_ans)
    match_score = (match_score.get("Perfect", 0) * 100)
    final = (sbert_score * 0.8) + (match_score * 0.2)  # Weighted combination
    return int(final)


async def parse_answer_key(input_text):
    pattern = re.findall(r"(\d+)\.\s(.*?)(?:\.(\d+))", input_text, re.DOTALL)
    result = []
    
    for match in pattern:
        question_number = int(match[0])
        model_answer = match[1].strip().replace("\n", " ")  # Removing extra newlines
        max_marks = int(match[2])
        
        result.append({
            "question_number": question_number,
            "model_answer": model_answer,
            "max_marks": max_marks
        })

    db_name = "answer_keys"
    # Insert data into the database
    await delete_all(db_name)
    await insert_to_db(db_name, result)


def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )
    probs = bert_model.predict(test_data[0])[0]

    labels_probs = {labels[i]: float(probs[i]) for i, _ in enumerate(labels)}
    return labels_probs


def my_ocr(image_input, type):
    try:
        if isinstance(image_input, Image.Image):
            temp_path = "temp_image.png"
            image_input.save(temp_path)  # Save image as file
            image_input = temp_path  # Use the file path instead of image object

        # Run OCR using the file path
        if type == "plain":
            res = ocr_model.chat(tokenizer, image_input, ocr_type='ocr')
        else:
            res = ocr_model.chat(tokenizer, image_input, ocr_type='format')

        return res
    except Exception as e:
        flash(f"An unexpected error occurred: {e}", "error")  # Ensure secret_key is set
        return ""



def evaluate_answers(student_answers):
    """
    Evaluates student answers against the answer key and assigns scaled marks.
    """
    questions = []
    examdata = {}
    total_marks = 0
    acured_mark = 0
    answer_key = asyncio.run(retrieve_all_answers("answer_keys"))
    for q_num, student_text in student_answers.items():
        if q_num in answer_key:
            answer_key_text = answer_key[q_num]["model_answer"]
            max_marks = answer_key[q_num]["max_marks"]
            total_marks += max_marks
            # Get similarity score
            similarity_scores = final_score(student_text, answer_key_text)

            # Extract similarity percentage
            perfect_match_score = similarity_scores

            # Convert similarity percentage to max marks
            scaled_marks = (perfect_match_score / 100) * max_marks
            acured_mark += round(scaled_marks, 1)
            feedback = get_answer_feedback(answer_key_text,student_text,perfect_match_score,acured_mark)
            # Store results
            questions.append({
                "id": q_num,
                "question_num": q_num,
                "student_ans": student_text,
                "model_ans": answer_key_text,
                "marks": round(scaled_marks, 1),
                "max_marks": max_marks,
                "feedback": feedback
            })
    examdata["totalmarks"] = total_marks
    examdata["accuredmarks"] = round(acured_mark, 1)
    examdata["result"] = questions

    return examdata


@app.route("/ans_key_check", methods=["POST"])
def ans_key_upload():
    if request.method == 'POST':
        if "answer_key" not in request.files:
            return "Upload answerkey first!", 400
        
        answer_key = request.files["answer_key"]
        key_path = os.path.join(UPLOAD_FOLDER, answer_key.filename)
        answer_key.save(key_path)
        if answer_key.filename.endswith(".pdf"):
            answer_key_text = extract_text_from_pdf(key_path)
        else:
            answer_key_text = my_ocr(key_path,"plain")
        asyncio.run(parse_answer_key(answer_key_text))
        return render_template('success.html')
    

def extract_text_from_pdf(pdf_path):
    """Extracts text directly from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text


def extract_text_from_handwritten_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""

    for i, img in enumerate(images):
        temp_image_path = f"temp_page_{i}.png"
        img.save(temp_image_path, "PNG")  # Save image temporarily
        text += my_ocr(temp_image_path, "format")
        text += "\n"
        os.remove(temp_image_path)  # Delete temporary file after processing
    return text


@app.route("/upload_check", methods=["POST"])
def upload():
    if request.method == 'POST':
        if "answer_paper" not in request.files:
            return "images not required!", 400

        answer_paper = request.files["answer_paper"]

        # Save files temporarily
        paper_path = os.path.join(UPLOAD_FOLDER, answer_paper.filename)

        answer_paper.save(paper_path)

        if answer_paper.filename.endswith(".pdf"):
            student_answer_text = extract_text_from_handwritten_pdf(paper_path)
        else:
            student_answer_text = my_ocr(paper_path,"format")

        pattern = r"(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)"  # Matches question numbers and answers

        matches = re.findall(pattern, student_answer_text, re.DOTALL)

        # Convert matches to dictionary
        student_answers = {int(q_num): ans.strip().replace("\n", " ") for q_num, ans in matches}

        results = evaluate_answers(student_answers)

        student_name = request.form.get("student_name")
        roll_num = request.form.get("roll_number")

        
        return render_template('result.html', results_data=results, name=student_name, roll_no=roll_num)




@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        student_ans = request.form.get('student_ans')
        model_ans = request.form.get('model_ans')
        print(student_ans)
        print(model_ans)
        text = check_similarity(student_ans, model_ans)
        print(text)

        con_val = int(text["Contradiction"] * 100)
        per_val = int(text["Perfect"]*100)
        neu_val = int(text["Neutral"]*100)

        dict = {}
        dict['Contradiction'] = con_val
        dict['Perfect'] = per_val
        dict['Neutral'] = neu_val
        dict['student_ans'] = student_ans
        dict['model_ans'] = model_ans
        dict['marks'] = per_val 
        return render_template('answers.html', dict=dict)
    return render_template("index.html")


@app.route("/result")
def result():
    return render_template("result.html")


@app.route("/chat", methods=["POST"])
def user_chat():
    try:
        data = request.json
        print("Received data:", data)  # Debugging

        user_message = data.get("message")
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        response = chat(user_message)
        return jsonify({"response": response})

    except Exception as e:
        print(f"❌ Error in /chat: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


# @app.route("/auth")
# def auth():
#     return render_template("auth.html")



# @app.route("/signup", methods=["POST"])
# async def signup():
#     """Signup a user."""
#     data = request.json
#     name, email, password = data.get("name"), data.get("email"), data.get("password")

#     if not name or not email or not password:
#         return jsonify({"error": "Missing fields"}), 400

#     result = await create_user(name, email, password)
#     if "error" in result:
#         return jsonify(result), 400

#     return jsonify(result), 201


# @app.route("/login", methods=["POST"])
# async def login():
#     """Login a user and return JWT."""
#     data = request.json
#     email, password = data.get("email"), data.get("password")

#     user = await get_user_by_email(email)
#     if not user:
#         return jsonify({"error": "User not found"}), 404

#     # bcrypt is not async, so run it in a separate thread
#     is_valid = await asyncio.to_thread(
#         bcrypt.checkpw, password.encode('utf-8'), user["password"]
#     )

#     if is_valid:
#         access_token = create_access_token(identity=email)
#         return jsonify({"access_token": access_token}), 200

#     return jsonify({"error": "Invalid credentials"}), 401


if __name__ == '__main__':
    app.run()
