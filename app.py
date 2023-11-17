from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load the trained model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.post("/predict")
async def predict(features: dict):
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    return {"prediction": prediction[0]}
