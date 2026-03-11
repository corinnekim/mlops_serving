# 1. 베이스 이미지 설정
FROM python:3.13-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 라이브러리 설치
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn joblib pydantic python-multipart

# 4. 모델 파일과 API 코드를 컨테이너 안으로 복사
COPY main.py .
COPY model.joblib .

# 5. 서버 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]