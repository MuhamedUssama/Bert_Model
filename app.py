from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os
import gdown

app = Flask(__name__)

FILE_ID = "1clUv6Sr9wpNp2p56-aZ7Xu6LP9f9guNr"
MODEL_PATH = "Bert_person.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model file from Google Drive...")
    gdown.cached_download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# model = BertForSequenceClassification.from_pretrained(
#     'bert-base-uncased',
#     num_labels=5,
#     problem_type="multi_label_classification"
# )

# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device)
# model.eval()

try:
    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=5,
        problem_type="multi_label_classification"
    )
    print("Loading model weights from Bert_person.pth...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Moving model to device...")
    model.to(device)
    print("Setting model to evaluation mode...")
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e

trait_columns = ['Openness(O)', 'Conscientiousness(C)', 'Extraversion(E)', 'Agreeableness(A)', 'Neuroticism(N)']

def analyze_personality(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()[0]

    probabilities = torch.sigmoid(torch.tensor(logits)).numpy() * 100
    result = {trait: f'{prob:.2f}%' for trait, prob in zip(trait_columns, probabilities)}
    return result

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        result = analyze_personality(text)
        
        max_trait = max(result, key=lambda x: float(result[x][:-1]))
        
        response = {
            'traits': result,
            'dominant_trait': max_trait
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  
    print(f"🚀 Starting Server on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)
