from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os
import gdown

app = Flask(__name__)

FILE_ID = "1clUv6Sr9wpNp2p56-aZ7Xu6LP9f9guNr"
MODEL_PATH = "Bert_person.pth"

try:
    print("📂 Files in directory before download:", os.listdir())
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model file from Google Drive...")
        gdown.cached_download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Model file was not downloaded correctly!")
    
    print("📂 Files in directory after download:", os.listdir())
except Exception as e:
    print(f"❌ Error during model download: {e}")

# تحديد الجهاز المتاح
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Running on: {device}")

# تحميل التوكنيزر
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("✅ Tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")

# تحميل الموديل
try:
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=5,
        problem_type="multi_label_classification"
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# أسماء السمات الشخصية
trait_columns = ["Openness(O)", "Conscientiousness(C)", "Extraversion(E)", "Agreeableness(A)", "Neuroticism(N)"]

# تحليل النص وتوقع السمات
def analyze_personality(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return {"error": "Invalid text input!"}

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()[0]

        probabilities = torch.sigmoid(torch.tensor(logits)).numpy() * 100
        result = {trait: f"{prob:.2f}%" for trait, prob in zip(trait_columns, probabilities)}
        return result
    except Exception as e:
        print(f"❌ Error in analyze_personality: {e}")
        return {"error": str(e)}

# API لتحليل الشخصية
@app.route("/analyze", methods=["POST"])
def analyze_text():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data["text"]
        result = analyze_personality(text)
        
        if "error" in result:
            return jsonify(result), 400
        
        max_trait = max(result, key=lambda x: float(result[x][:-1]))

        response = {
            "traits": result,
            "dominant_trait": max_trait
        }
        
        return jsonify(response), 200
    except Exception as e:
        print(f"❌ Error in /analyze endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Health Check
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running"}), 200

# تشغيل السيرفر
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))  
        print(f"🚀 Starting Server on port {port}...")
        print("🌍 Environment Variables:", os.environ)  # Debugging
        app.run(debug=False, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"❌ Error while starting server: {e}")
