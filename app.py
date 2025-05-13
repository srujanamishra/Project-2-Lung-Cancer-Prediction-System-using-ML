from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the correct regression model
model = pickle.load(open("lung_cancer_model.pkl", "rb"))

# Initialize FastAPI
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing; restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# Match these exactly to your HTML form input fields
class PatientData(BaseModel):
    age: int
    gender: int
    smoking: int
    finger_discoloration: int
    mental_stress: int
    exposure_to_pollution: int
    long_term_illness: int
    energy_level: int
    immune_weakness: int
    breathing_issue: int
    alcohol_consumption: int
    throat_discomfort: int
    oxygen_saturation: float
    chest_tightness: int
    family_history: int
    smoking_family_history: int
    stress_immune: int

@app.post("/predict")
def predict(data: PatientData):
    features = np.array([[ 
        data.age,
        data.gender,
        data.smoking,
        data.finger_discoloration,
        data.mental_stress,
        data.exposure_to_pollution,
        data.long_term_illness,
        data.energy_level,
        data.immune_weakness,
        data.breathing_issue,
        data.alcohol_consumption,
        data.throat_discomfort,
        data.oxygen_saturation,
        data.chest_tightness,
        data.family_history,
        data.smoking_family_history,
        data.stress_immune
    ]])
    
    prediction = model.predict(features)[0]
    result = "Yes" if prediction == 1 else "No"
    return {"prediction": result}

