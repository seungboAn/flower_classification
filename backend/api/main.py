import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from enum import Enum
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import time
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

class PredictionRequest(BaseModel):
    image: str  # base64 encoded image

class PredictionResponse(BaseModel):
    species: str
    confidence: float
    inference_time: float

# 꽃 종류 매핑
FLOWER_SPECIES = {0: 'dandelion', 1: 'daisy', 2: 'tulips', 3: 'sunflowers', 4: 'roses'}

# Load models
MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), 'models')

models = {
    ModelOption.base: tf.keras.models.load_model(os.path.join(MODEL_DIR, 'base.h5')),
}

async def preprocess_image(image_file: UploadFile, target_size: tuple = (224, 224)) -> np.ndarray:
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, 0)
    
    return image_array

@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...), option: ModelOption = Form(ModelOption.base)):
    # 사용된 옵션 로깅
    logger.info(f"Prediction requested with option: {option}")
    
    # 이미지 전처리
    preprocessed_image = await preprocess_image(image)
    
    # 모델 선택 및 예측
    model = models[option]
    
    # 추론 시간 측정 시작
    start_time = time.time()
    
    # 예측 수행
    prediction = model.predict(preprocessed_image)
    
    # 추론 시간 측정 종료
    inference_time = time.time() - start_time
    
    # prediction 리스트 로깅
    logger.info(f"Raw prediction: {prediction[0].tolist()}")
    
    # 예측 결과 해석
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class])
    
    # 해파리 종류 매핑 (예측된 클래스 인덱스를 실제 종 이름으로 변환)
    species = FLOWER_SPECIES.get(predicted_class, "Unknown")

    # 결과 로깅
    logger.info(f"Prediction result: species={species}, confidence={confidence:.4f}, inference_time={inference_time:.4f}")
    
    return PredictionResponse(
        species=species,
        confidence=confidence,
        inference_time=inference_time
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)