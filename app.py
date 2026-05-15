from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)

model = joblib.load("news_model.pkl")

@app.route('/')
def home():
    return "News Classifier API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)   # ⭐ FIX (JSON support)

    news = data['news'].lower().strip()

    prediction = model.predict([news])[0]

    return jsonify({
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)