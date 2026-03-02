from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import os

app = Flask(__name__)
CORS(app)

# Load model saat startup
MODEL_PATH = os.getenv('MODEL_PATH', 'glucose_model_v3.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Dictionary untuk menyimpan histori per user
user_histories = defaultdict(lambda: deque(maxlen=10))
JUMLAH_SAMPEL = 10

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'Glucose Prediction API',
        'version': '1.0',
        'endpoints': {
            'predict': '/predict [POST]',
            'health': '/health [GET]',
            'clear': '/clear/<user_id> [DELETE]'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        
        # Validasi input
        required_fields = ['user_id', 'ir', 'red', 'bpm']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        user_id = data['user_id']
        ir = float(data['ir'])
        red = float(data['red'])
        bpm = float(data['bpm'])
        
        # Validasi nilai
        if ir <= 0 or red <= 0 or bpm <= 0:
            return jsonify({'error': 'Invalid sensor values'}), 400
        
        # Prediksi
        input_df = pd.DataFrame([{
            'IR': ir,
            'RED': red,
            'BPM': bpm
        }])
        
        prediction = model.predict(input_df)[0]
        
        # Tambahkan ke histori user
        user_histories[user_id].append(prediction)
        current_count = len(user_histories[user_id])
        
        response = {
            'user_id': user_id,
            'prediction': float(prediction),
            'sample_count': current_count,
            'max_samples': JUMLAH_SAMPEL,
            'ready': False
        }
        
        # Jika sudah 10 sampel, hitung rata-rata
        if current_count == JUMLAH_SAMPEL:
            average = int(round(np.mean(list(user_histories[user_id]))))
            response['final_prediction'] = average
            response['ready'] = True
            
            # Tentukan status
            if average < 70:
                response['status'] = 'Low'
                response['status_id'] = 'Rendah'
            elif average <= 140:
                response['status'] = 'Normal'
                response['status_id'] = 'Normal'
            else:
                response['status'] = 'High'
                response['status_id'] = 'Cukup Tinggi'
            
            # Clear histori setelah prediksi final
            user_histories[user_id].clear()
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({'error': f'Invalid data type: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/clear/<user_id>', methods=['DELETE'])
def clear_history(user_id):
    """Manual clear histori untuk user tertentu"""
    if user_id in user_histories:
        user_histories[user_id].clear()
        return jsonify({'message': f'History cleared for {user_id}'}), 200
    return jsonify({'message': 'No history found'}), 404

@app.route('/status/<user_id>', methods=['GET'])
def get_status(user_id):
    """Cek status histori user"""
    count = len(user_histories.get(user_id, []))
    return jsonify({
        'user_id': user_id,
        'sample_count': count,
        'max_samples': JUMLAH_SAMPEL,
        'samples_needed': JUMLAH_SAMPEL - count
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)