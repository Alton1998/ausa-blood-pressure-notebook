import os

import numpy as np
from fastapi import FastAPI
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import os

app = FastAPI()
print(os.curdir)
model = load_model("bp.keras")


class BPVitals(BaseModel):
    age: float
    sex: int
    bmi: float
    systolic_bp: list[float]
    diastolic_bp: list[float]


@app.post("/bp")
async def predict_bp(bp_vitals: BPVitals):
    context_data = [bp_vitals.age, bp_vitals.sex, bp_vitals.bmi]
    time_series_data = [bp_vitals.systolic_bp, bp_vitals.diastolic_bp]
    context_data = np.array(context_data, dtype=float)
    time_series_data = np.array(time_series_data, dtype=float).reshape(-1, 14, 2)
    results = model.predict(x=[context_data, time_series_data])
    return {"future_systolic_bp": results[0:7], "future_diastolic_bp": results[7:14]}
