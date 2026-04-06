from flask import Flask, request, jsonify
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'crop_model.pkl'))
le    = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

with open(os.path.join(BASE_DIR, 'metadata.json')) as f:
    metadata = json.load(f)

FEATURES = metadata['features']
CLASSES  = metadata['classes']

CROP_EMOJI = {
    'rice': '🌾', 'maize': '🌽', 'chickpea': '🫘', 'kidneybeans': '🫘',
    'pigeonpeas': '🫘', 'mothbeans': '🫘', 'mungbean': '🫘', 'blackgram': '🫘',
    'lentil': '🌿', 'pomegranate': '🍎', 'banana': '🍌', 'mango': '🥭',
    'grapes': '🍇', 'watermelon': '🍉', 'muskmelon': '🍈', 'apple': '🍎',
    'orange': '🍊', 'papaya': '🍈', 'coconut': '🥥', 'cotton': '🌸',
    'jute': '🌿', 'coffee': '☕',
}


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        values = []
        for feat in FEATURES:
            if feat not in data:
                return jsonify({'error': f'Missing field: {feat}'}), 400
            values.append(float(data[feat]))

        X = np.array([values])

        pred_encoded = model.predict(X)[0]
        pred_proba   = model.predict_proba(X)[0]
        crop_name    = le.inverse_transform([pred_encoded])[0]
        confidence   = round(float(pred_proba[pred_encoded]) * 100, 2)

        top3_idx = pred_proba.argsort()[-3:][::-1]
        top3 = [
            {
                'crop':        le.inverse_transform([i])[0],
                'probability': round(float(pred_proba[i]) * 100, 2),
                'emoji':       CROP_EMOJI.get(le.inverse_transform([i])[0], '🌱')
            }
            for i in top3_idx
        ]

        return jsonify({
            'success':           True,
            'recommended_crop':  crop_name,
            'emoji':             CROP_EMOJI.get(crop_name, '🌱'),
            'confidence':        confidence,
            'top3':              top3,
            'model_accuracy':    metadata['accuracy'] * 100,
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status':           'ok',
        'model':            metadata['model_name'],
        'accuracy':         metadata['accuracy'],
        'crops_supported':  len(CLASSES)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
