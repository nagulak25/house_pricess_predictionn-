import pickle
import pandas as pd

def load_model():
    with open('models/model.pkl', 'rb') as f:
        return pickle.load(f)

def predict(input_data: dict):
    model = load_model()

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]

    return prediction