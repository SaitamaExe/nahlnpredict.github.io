from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Memuat model yang telah dilatih
with open('DTRModel.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        features = [
            float(data['lebar_bangunan']),
            float(data['lebar_tanah']),
            float(data['jumlah_kamar_tidur']),
            float(data['jumlah_kamar_mandi']),
            float(data['jumlah_kapasitas_mobil_dan_garasi'])
        ]
        print(f"Features: {features}")  # Debugging: Print input features

        # Melakukan prediksi
        prediction = model.predict([features])[0]
        print(f"Prediction: {prediction}")  # Debugging: Print the prediction

        # Membuat representasi gabungan dari fitur
        combined_features = {
            'Lebar Bangunan': data['lebar_bangunan'],
            'Lebar Tanah': data['lebar_tanah'],
            'Jumlah Kamar Tidur': data['jumlah_kamar_tidur'],
            'Jumlah Kamar Mandi': data['jumlah_kamar_mandi'],
            'Jumlah Kapasitas Mobil dan Garasi': data['jumlah_kapasitas_mobil_dan_garasi']
        }

        # Menyiapkan hasil
        result = {
            'Combined Features': combined_features,
            'Prediction': f"Rp {prediction:,.2f}"
        }
        print(f"Result: {result}")  # Debugging: Print the result
        return render_template('result.html', result=result)
    except ValueError as e:
        print(f"Value Error: {e}")
        return "Invalid input. Please ensure all values are numeric."

if __name__ == '__main__':
    app.run(debug=True)
