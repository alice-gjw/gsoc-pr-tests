from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

model = None


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    predicted_class: int
    class_name: str


CLASS_NAMES = ["setosa", "versicolor", "virginica"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    import pathlib
    model = joblib.load(pathlib.Path(__file__).parent / "data/model.joblib")
    yield


app = FastAPI(title="Iris Inference Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    prediction = model.predict([req.features])[0]
    return PredictResponse(
        predicted_class=int(prediction),
        class_name=CLASS_NAMES[int(prediction)],
    )
