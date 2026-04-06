from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from scripts.inference import predict_letter

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    letter, confidence = predict_letter(frame)

    return {
        "prediction": letter,
        "confidence": confidence
    }