from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import joblib
import os

app = Flask(__name__)
app.secret_key = 'secret'

# Globals
model = None
df_info = {}

MODEL_PATH = "models/best_model.pkl"

# Fixed feature schema based on house_prices.csv
NUM_FEATURES = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
CAT_FEATURES = ['Location', 'Property_Type']
CAT_VALUES = {
    'Location': ['City Center', 'Rural', 'Suburb'],
    'Property_Type': ['Apartment', 'House', 'Villa']
}
TARGET = 'Price'


@app.route('/')
def home():
    return render_template(
        'index.html',
        model_trained=(model is not None),
        num_features=NUM_FEATURES,
        cat_features=CAT_FEATURES,
        cat_values=CAT_VALUES,
        df_info=df_info
    )


@app.route('/train', methods=['POST'])
def train_model():
    global model, df_info

    try:
        df = pd.read_csv('house_prices.csv')

        # Drop identifier column
        df = df.drop(columns=['Property_ID'], errors='ignore')

        df_info['shape'] = df.shape
        df_info['target'] = TARGET

        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Pipeline
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), NUM_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES)
        ])

        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge()
        }

        best_score = -1
        best_model = None
        best_name = ""

        for name, m in models.items():
            pipe = Pipeline([
                ('prep', preprocessor),
                ('model', m)
            ])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            score = r2_score(y_test, pred)

            if score > best_score:
                best_score = score
                best_model = pipe
                best_name = name

        model = best_model

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        df_info['r2_score'] = best_score
        df_info['best_model'] = best_name

        return jsonify({
            "status": "success",
            "message": f"{best_name} trained successfully (R²={best_score:.3f})"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    global model

    try:
        if model is None:
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
            else:
                return jsonify({"status": "error", "message": "Please train the model first."})

        data = request.json

        # Cast numeric fields
        for col in NUM_FEATURES:
            if col in data:
                data[col] = float(data[col])

        df_input = pd.DataFrame([data])

        prediction = model.predict(df_input)[0]

        return jsonify({
            "status": "success",
            "predicted_price": float(prediction),
            "confidence_low": float(prediction * 0.9),
            "confidence_high": float(prediction * 1.1)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    print("🚀 Running on http://127.0.0.1:5000")
    app.run(debug=True)