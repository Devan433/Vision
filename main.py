"""
Object Detection REST API
=========================
A full-stack computer vision API built with FastAPI and YOLOv8.
Upload any image and get detected objects with bounding boxes,
class labels, and confidence scores.

Author: Deva Nandan H
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
from typing import Optional


# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="Object Detection API",
    description=(
        "Upload an image and get real-time object detection results. "
        "Powered by YOLOv8 and FastAPI."
    ),
    version="1.0.0",
    contact={
        "name": "Deva Nandan H",
        "email": "devanandan0123@gmail.com",
    },
)

# ============================================================
# MODEL LOADING (loaded once at startup, not per request)
# ============================================================

print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")  # Downloads automatically on first run (~6MB)
print("Model loaded successfully!")

# ============================================================
# DIRECTORY SETUP
# ============================================================

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Serve annotated result images as static files
app.mount("/static/results", StaticFiles(directory=RESULTS_DIR), name="results")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


def validate_image(file: UploadFile) -> None:
    """Validate uploaded file is an allowed image type."""
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file type",
                "received": file.content_type,
                "allowed": list(ALLOWED_EXTENSIONS),
            },
        )


def decode_image(contents: bytes) -> np.ndarray:
    """Decode raw bytes into an OpenCV image (NumPy array)."""
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. File may be corrupted.",
        )
    return img


def run_detection(img: np.ndarray, confidence: float) -> dict:
    """
    Run YOLOv8 inference on an image.
    
    Returns a dict with detection results and the annotated image.
    """
    results = model(img, conf=confidence, verbose=False)

    detections = []
    class_counts = {}

    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            conf = round(float(box.conf), 3)
            coords = box.xyxy[0].tolist()

            detection = {
                "class": class_name,
                "confidence": conf,
                "bounding_box": {
                    "x1": int(coords[0]),
                    "y1": int(coords[1]),
                    "x2": int(coords[2]),
                    "y2": int(coords[3]),
                    "width": int(coords[2] - coords[0]),
                    "height": int(coords[3] - coords[1]),
                },
            }
            detections.append(detection)

            # Count objects per class
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Sort detections by confidence (highest first)
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    # Get annotated image with bounding boxes drawn by YOLO
    annotated_img = results[0].plot()

    return {
        "detections": detections,
        "class_counts": class_counts,
        "annotated_img": annotated_img,
    }


# ============================================================
# API ENDPOINTS
# ============================================================


@app.get("/", tags=["Health"])
def health_check():
    """Health check — verify the API and model are running."""
    return {
        "status": "running",
        "model": "YOLOv8n (nano)",
        "model_classes": len(model.names),
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "detect": "POST /detect — Upload an image for object detection",
            "classes": "GET /classes — List all detectable object classes",
            "docs": "GET /docs — Interactive API documentation",
        },
    }


@app.get("/classes", tags=["Info"])
def list_classes():
    """List all 80 object classes that YOLOv8 can detect."""
    return {
        "total_classes": len(model.names),
        "classes": model.names,
    }


@app.post("/detect", tags=["Detection"])
async def detect_objects(
    file: UploadFile = File(..., description="Image file (jpg, png, webp)"),
    confidence: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold (0.0 to 1.0)",
    ),
):
    """
    Upload an image and detect objects.

    Returns:
    - List of detected objects with class, confidence, and bounding box
    - Summary with object counts per class
    - URL to the annotated image with bounding boxes drawn
    """

    # --- VALIDATE INPUT ---
    validate_image(file)

    # --- READ & DECODE IMAGE ---
    contents = await file.read()
    img = decode_image(contents)
    height, width, channels = img.shape

    # --- RUN DETECTION ---
    result = run_detection(img, confidence)

    # --- SAVE ANNOTATED IMAGE ---
    result_id = str(uuid.uuid4())[:8]
    result_filename = f"{result_id}.jpg"
    result_path = os.path.join(RESULTS_DIR, result_filename)
    cv2.imwrite(result_path, result["annotated_img"])

    # --- RETURN RESPONSE ---
    return {
        "image_info": {
            "filename": file.filename,
            "width": width,
            "height": height,
            "channels": channels,
        },
        "detection_config": {
            "model": "YOLOv8n",
            "confidence_threshold": confidence,
        },
        "summary": {
            "total_objects_detected": len(result["detections"]),
            "objects_per_class": result["class_counts"],
        },
        "detections": result["detections"],
        "annotated_image_url": f"/static/results/{result_filename}",
    }


@app.post("/detect/image", tags=["Detection"])
async def detect_and_return_image(
    file: UploadFile = File(..., description="Image file (jpg, png, webp)"),
    confidence: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    ),
):
    """
    Upload an image and get back the annotated image directly
    (with bounding boxes drawn).
    """

    validate_image(file)
    contents = await file.read()
    img = decode_image(contents)

    result = run_detection(img, confidence)

    # Save and return the annotated image
    result_id = str(uuid.uuid4())[:8]
    result_path = os.path.join(RESULTS_DIR, f"{result_id}.jpg")
    cv2.imwrite(result_path, result["annotated_img"])

    return FileResponse(
        result_path,
        media_type="image/jpeg",
        filename=f"detected_{file.filename}",
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("  Object Detection API")
    print("  Server: http://localhost:8000")
    print("  Docs:   http://localhost:8000/docs")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
