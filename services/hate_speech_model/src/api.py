from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

from src.model import HateSpeechClassifier


class Comment(BaseModel):
    id: int
    comment_text: str


class CommentRequest(BaseModel):
    data: List[Comment]

    class Config:
        # this will be used as the example in Swagger docs
        schema_extra = {
            "example": {
                "data": [
                    {
                        "id": 1,
                        "comment_text": "you are so fucking stupid",
                    },
                    {
                        "id": 2,
                        "comment_text": "as eloquent as your comment was, i respectfully disagree",
                    }
                ],
            }
        }


class Prediction(BaseModel):
    id: int
    comment_text: str
    pred: float


class PredictionResponse(BaseModel):
    preds: List[Prediction]


cls = HateSpeechClassifier()
cls.load_classifier(load_path="src/models/")
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
Instrumentator().instrument(app).expose(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# start app with: uvicorn api:app --reload
# to view the app: http://localhost:8000
# to view metrics: http://localhost:8000/metrics
# when running with Prometheus, configure a job for this app in prometheus.yml and test at http://localhost:9090/targets


@app.post("/")
@limiter.limit("1/second")
def classify(data: CommentRequest, request: Request) -> PredictionResponse:
    preds = cls.predict(jsonable_encoder(data.data))
    return PredictionResponse(preds=preds)
