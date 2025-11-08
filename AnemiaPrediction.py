from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load your saved Random Forest model
model = joblib.load('anemia_rf_model.pkl')

class AnemiaInput(BaseModel):   
    Hemoglobin: float
    MCH: float
    MCHC: float
    MCV: float

@app.post("/predict")
def predict_anemia(data: AnemiaInput):
    features = np.array([[data.Hemoglobin, data.MCH, data.MCHC, data.MCV]])
    prediction = model.predict(features)[0]
    return {"prediction": "Anemic" if prediction == 1 else "Normal"}
