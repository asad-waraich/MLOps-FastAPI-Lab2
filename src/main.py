from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data
import numpy as np

app = FastAPI()

# 1. Define the input data model with 4 features
class SyntheticData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# 2. Define the response model
class PredictionResponse(BaseModel):
    prediction: int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: SyntheticData):
    try:
        # 3. Convert the input data into a numpy array for the model
        feature_list = [[
            features.feature1,
            features.feature2,
            features.feature3,
            features.feature4
        ]]
        
        prediction_result = predict_data(feature_list)
        
        # 4. Return the prediction in the correct response format
        return PredictionResponse(prediction=int(prediction_result[0]))
    
    except Exception as e:
        # If anything goes wrong, return a 500 error
        raise HTTPException(status_code=500, detail=str(e))