"""uvicorn app:app --reload"""

from fastapi import FastAPI 
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("data/model/lr_pipeline.joblib")
embedder = model["embedder"]
pipeline = model["pipeline"]
categories = model["categories"]

class Request(BaseModel):
    text: str 
    
@app.get("/")
def root():
    return {"message": "20 Newsgroups classifier API"}
    
@app.post("/predict")
def predict(req: Request):
    vector = embedder.encode([req.text])
    prediction = pipeline.predict(vector)[0]
    return {"category": categories[prediction]}

