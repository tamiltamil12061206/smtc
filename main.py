from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load models and vectorizer
price_model = joblib.load("price_model.pkl")
quantity_model = joblib.load("quantity_model.pkl")
uses_model = joblib.load("uses_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
le_uses = joblib.load("le_uses.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    fertilizer_name = data.get("name")
    
    if not fertilizer_name:
        return jsonify({"error": "No fertilizer name provided"}), 400
    
    # Vectorize the input
    name_vectorized = vectorizer.transform([fertilizer_name])
    
    # Predict values
    predicted_price = price_model.predict(name_vectorized)[0]
    predicted_quantity = quantity_model.predict(name_vectorized)[0]
    predicted_uses_encoded = uses_model.predict(name_vectorized)[0]
    predicted_uses = le_uses.inverse_transform([predicted_uses_encoded])[0]
    
    result = {
        "Uses": predicted_uses,
        "Price": round(predicted_price, 2),
        "Quantity": round(predicted_quantity, 2)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
