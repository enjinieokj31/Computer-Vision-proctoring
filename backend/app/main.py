from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import asyncio

from app.model import YOLOModel

app = FastAPI(title="YOLO Face Detection API")

allow_origins = [os.getenv("FRONTEND_URL")]
# Allow frontend later
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def maintenance_guard(request, call_next):
    if os.getenv("MAINTENANCE_MODE") == "true":
        if request.url.path != "/health":
            return JSONResponse(
                status_code=503,
                content={"message": "Maintenance in progress"}
            )

    response = await call_next(request)
    return response

# Load model once
model = YOLOModel("weights/model.onnx")

# Only 1 inference at a time
lock = asyncio.Lock()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image upload allowed")

    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    async with lock:
        detections, latency = model.predict(image)

    face_count = len(detections)

    return {
        "message": "success",
        "latency_ms": latency,
        "face_count": face_count,
        "multiple_faces": face_count > 1
    }