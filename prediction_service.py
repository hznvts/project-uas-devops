from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model & encoders
model = joblib.load("model.pkl")
le_carrier = joblib.load("le_carrier.pkl")
le_origin = joblib.load("le_origin.pkl")
le_dest = joblib.load("le_dest.pkl")

app = FastAPI()

class FlightData(BaseModel):
    carrier: str
    origin: str
    dest: str
    distance: int

@app.post("/predict")
def predict(data: FlightData):
    x_input = pd.DataFrame({
        'OP_CARRIER_enc': [le_carrier.transform([data.carrier])[0]],
        'ORIGIN_enc': [le_origin.transform([data.origin])[0]],
        'DEST_enc': [le_dest.transform([data.dest])[0]],
        'DISTANCE': [data.distance]
    })
    pred = model.predict(x_input)[0]
    return {"prediction": int(pred)}
