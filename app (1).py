"""
Crop Recommendation System — Flask Backend
==========================================
1. Place your exported model files in the same directory:
   - crop_model.pkl
   - label_encoder.pkl
   - metadata.json

2. Install dependencies:
   pip install flask scikit-learn joblib numpy

3. Run:
   python app.py
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

# ─── Load model artifacts ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model    = joblib.load(os.path.join(BASE_DIR, 'crop_model.pkl'))
le       = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

with open(os.path.join(BASE_DIR, 'metadata.json')) as f:
    metadata = json.load(f)

FEATURES = metadata['features']   # ['N','P','K','temperature','humidity','ph','rainfall']
CLASSES  = metadata['classes']

# ─── Crop emoji map ──────────────────────────────────────────────────────────
CROP_EMOJI = {
    'rice': '🌾', 'maize': '🌽', 'chickpea': '🫘', 'kidneybeans': '🫘',
    'pigeonpeas': '🫘', 'mothbeans': '🫘', 'mungbean': '🫘', 'blackgram': '🫘',
    'lentil': '🌿', 'pomegranate': '🍎', 'banana': '🍌', 'mango': '🥭',
    'grapes': '🍇', 'watermelon': '🍉', 'muskmelon': '🍈', 'apple': '🍎',
    'orange': '🍊', 'papaya': '🍈', 'coconut': '🥥', 'cotton': '🌸',
    'jute': '🌿', 'coffee': '☕',
}

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the frontend HTML (embed it here for single-file deployment)."""
    with open(os.path.join(BASE_DIR, 'index.html')) as f:
        html = f.read()
    return html


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body (JSON): { "N": 90, "P": 42, "K": 43, "temperature": 21,
                   "humidity": 82, "ph": 6.5, "rainfall": 210 }
    """
    try:
        data = request.get_json(force=True)

        # Validate and build feature vector
        values = []
        for feat in FEATURES:
            if feat not in data:
                return jsonify({'error': f'Missing field: {feat}'}), 400
            values.append(float(data[feat]))

        X = np.array([values])

        # Predict
        pred_encoded    = model.predict(X)[0]
        pred_proba      = model.predict_proba(X)[0]
        crop_name       = le.inverse_transform([pred_encoded])[0]
        confidence      = round(float(pred_proba[pred_encoded]) * 100, 2)

        # Top-3 recommendations
        top3_idx = pred_proba.argsort()[-3:][::-1]
        top3 = [
            {
                'crop': le.inverse_transform([i])[0],
                'probability': round(float(pred_proba[i]) * 100, 2),
                'emoji': CROP_EMOJI.get(le.inverse_transform([i])[0], '🌱')
            }
            for i in top3_idx
        ]

        return jsonify({
            'success': True,
            'recommended_crop': crop_name,
            'emoji': CROP_EMOJI.get(crop_name, '🌱'),
            'confidence': confidence,
            'top3': top3,
            'model_accuracy': metadata['accuracy'] * 100,
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': metadata['model_name'],
        'accuracy': metadata['accuracy'],
        'crops_supported': len(CLASSES)
    })


if __name__ == '__main__':
    print(f"🌾 Crop Recommendation API running...")
    print(f"   Model  : {metadata['model_name']}")
    print(f"   Accuracy: {metadata['accuracy']*100:.2f}%")
    print(f"   Crops  : {len(CLASSES)}")
    app.run(debug=True, port=5000)
