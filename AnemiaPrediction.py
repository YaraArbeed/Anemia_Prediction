from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware   
import joblib
import numpy as np

app = FastAPI()

origins = [
    "http://localhost:5173"   # your local frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)
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
