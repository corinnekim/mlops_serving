from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

# 1. FastAPI 서비스 객체를 생성하여 API 통신을 준비합니다.
app = FastAPI()

# 2. joblib을 사용하여 사전에 학습된 사이킷런 모델 파일을 메모리로 불러옵니다.
model = joblib.load('model.joblib')

# 3. 클라이언트가 보내야 할 데이터의 구조와 타입(실수 리스트)을 정의합니다.
class PredictRequest(BaseModel):
    data: List[float]

# 4. 브라우저에서 바로 확인하는 용도 (http://IP주소/)
@app.get("/")
def home():
    return {"message": "CI/CD 자동화 파이프라인 성공"}

# 5. 붓꽃 클래스 예측용 용도 (http://IP주소/predict)
@app.post("/predict")
def predict(request: PredictRequest):
    prediction = model.predict([request.data])
    return {"class_index": int(prediction[0])}