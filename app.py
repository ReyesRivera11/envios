from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import pandas as pd

app = Flask(__name__)

# CORS solo en rutas específicas
cors_config = {
    "origins": ["http://localhost:5173","https://cayro.netlify.app"],  # o usa ["http://localhost:5173", "https://tudominio.com"] para mayor seguridad
    "methods": ["POST"],
    "allow_headers": ["Content-Type"]
}

# === Cargar modelos entrenados ===
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("random_forest_model.pkl")

# === Ruta simple para probar si el backend está vivo ===
@app.route("/ping", methods=["GET"])
@cross_origin(**cors_config)
def ping():
    return jsonify({"message": "pong"}), 200

# === Ruta principal para predicción ===
@app.route("/predict", methods=["POST"])
@cross_origin(**cors_config)
def predict():
    try:
        data = request.get_json()

        required_fields = ["subtotalAmount", "totalAmount", "num_items", "total_quantity", "state"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Faltan campos requeridos"}), 400

        df = pd.DataFrame([{
            "subtotalAmount": float(data["subtotalAmount"]),
            "totalAmount": float(data["totalAmount"]),
            "num_items": int(data["num_items"]),
            "total_quantity": int(data["total_quantity"]),
            "state": data["state"]
        }])

        df[["state"]] = encoder.transform(df[["state"]])
        X_scaled = scaler.transform(df)
        prediction = round(model.predict(X_scaled)[0], 2)

        return jsonify({"shipping_cost_prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
