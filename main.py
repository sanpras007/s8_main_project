import tensorflow as tf
import transformers
import asyncio
from flask import Flask,flash, render_template, request
from DB_operations import insert_to_db, delete_all, retrieve_all_answers
from database import client
import numpy as np
from huggingface_hub import from_pretrained_keras
import tensorflow_hub as hub
from transformers import AutoModel, AutoTokenizer
import os as os
import re
from motor.motor_asyncio import AsyncIOMotorClient

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db = client["database_name"]
collection = db["collection_name"]


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
    return render_template("demo.html")


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


model = from_pretrained_keras("keras-io/bert-semantic-similarity")
# model = tf.keras.models.load_model('saved_model_pooja.h5')
# model = tf.keras.models.load_model(
#        ('saved_model_pooja.h5'),
#        custom_objects={'TFBertMainLayer':hub.TFBertMainLayer}
# )
labels = ["Contradiction", "Perfect", "Neutral"]


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
    probs = model.predict(test_data[0])[0]

    labels_probs = {labels[i]: float(probs[i]) for i, _ in enumerate(labels)}
    return labels_probs

    # idx = np.argmax(proba)
    # proba = f"{proba[idx]*100:.2f}%"
    # pred = labels[idx]
    # return f'The semantic similarity of two input sentences is {pred} with {proba} of probability'

def my_ocr(image_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
        model = model.eval().cuda()
        res = model.chat(tokenizer, image_path, ocr_type='format')

        return res
    except Exception as e:
        flash(f"An unexpected error occurred: {e}")



def extract_answers(ocr_text):
    """
    Extracts answers from OCR text and maps them to question numbers.
    
    Args:
        ocr_text (str): The extracted text from OCR.
    
    Returns:
        dict: A dictionary mapping question numbers to their respective answers.
    """
    # Use regex to find question numbers followed by answers
    pattern = r"(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)"  # Matches question numbers and answers

    matches = re.findall(pattern, ocr_text, re.DOTALL)

    # Convert matches to dictionary
    student_answers = {int(q_num): ans.strip().replace("\n", " ") for q_num, ans in matches}

    evaluate_answers(student_answers)

def evaluate_answers(student_answers):
    """
    Evaluates student answers against the answer key and assigns scaled marks.
    """
    results = {}
    answer_key = asyncio.run(retrieve_all_answers("answer_keys"))
    for q_num, student_text in student_answers.items():
        if q_num in answer_key:
            answer_key_text = answer_key[q_num]["model_answer"]
            max_marks = answer_key[q_num]["max_marks"]

            # Get similarity score
            similarity_scores = check_similarity(student_text, answer_key_text)

            # Extract similarity percentage
            perfect_match_score = int(similarity_scores["Perfect"] * 100)

            # Convert similarity percentage to max marks
            scaled_marks = (perfect_match_score / 100) * max_marks

            # Store results
            results[q_num] = {
                "question": q_num,
                "student_ans": student_text,
                "model_ans": answer_key_text,
                "marks": round(scaled_marks, 2),
                "max_marks": max_marks
            }

    return results


@app.route("/ans_key_check", methods=["POST"])
def ans_key_upload():
    if request.method == 'POST':
        if "answer_key" not in request.files:
            return "Upload answerkey first!", 400
        
        answer_key = request.files["answer_key"]
        key_path = os.path.join(UPLOAD_FOLDER, answer_key.filename)
        answer_key.save(key_path)
        answer_key_text = my_ocr(key_path)
        asyncio.run(parse_answer_key(answer_key_text))
        return render_template('success.html')



@app.route("/upload_check", methods=["POST"])
def upload():
    if request.method == 'POST':
        if "answer_paper" not in request.files:
            return "Both images are required!", 400

        answer_paper = request.files["answer_paper"]

        # Save files temporarily
        paper_path = os.path.join(UPLOAD_FOLDER, answer_paper.filename)

        answer_paper.save(paper_path)

        # Perform OCR on both images
        student_answer_text = my_ocr(paper_path)

        pattern = r"(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)"  # Matches question numbers and answers

        matches = re.findall(pattern, student_answer_text, re.DOTALL)

        # Convert matches to dictionary
        student_answers = {int(q_num): ans.strip().replace("\n", " ") for q_num, ans in matches}

        results = evaluate_answers(student_answers)
        
        return render_template('answers.html', dict=results)




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


if __name__ == '__main__':
    app.run()
