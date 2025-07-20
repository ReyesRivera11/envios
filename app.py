from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# === Cargar modelos ===
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("random_forest_model.pkl")

# === Ruta principal para predicción ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Verificar campos requeridos
        required_fields = ["subtotalAmount", "totalAmount", "num_items", "total_quantity", "state"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Faltan campos requeridos"}), 400

        # Crear DataFrame
        df = pd.DataFrame([{
            "subtotalAmount": float(data["subtotalAmount"]),
            "totalAmount": float(data["totalAmount"]),
            "num_items": int(data["num_items"]),
            "total_quantity": int(data["total_quantity"]),
            "state": data["state"]
        }])

        # Codificar
        df[["state"]] = encoder.transform(df[["state"]])

        # Escalar
        X_scaled = scaler.transform(df)

        # Predecir
        prediction = round(model.predict(X_scaled)[0], 2)

        return jsonify({"shipping_cost_prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
