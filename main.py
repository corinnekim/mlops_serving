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

# 4. '/predict' 주소로 데이터(POST)가 들어오면 실행될 경로를 설정합니다.
@app.post("/predict")
def predict(request: PredictRequest):
    # 5. 입력받은 데이터를 모델이 인식할 수 있는 행렬 형태로 변환하여 예측합니다.
    prediction = model.predict([request.data])
    
    # 6. 예측된 결과값(정수)을 JSON 형식의 딕셔너리로 최종 반환합니다.
    return {"class_index": int(prediction[0])}