# Object Detection REST API

A full-stack computer vision API built with **FastAPI** and **YOLOv8**. Upload any image and get real-time object detection results with bounding boxes, class labels, and confidence scores.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | Python, FastAPI, Uvicorn |
| **CV Model** | YOLOv8n (Ultralytics) — 80 object classes |
| **Image Processing** | OpenCV, NumPy |
| **Input Validation** | Pydantic (via FastAPI) |

## Features

- **Real-time object detection** — Upload any image, get detected objects instantly
- **Structured JSON response** — Class labels, confidence scores, bounding box coordinates
- **Annotated image output** — Get the image back with bounding boxes drawn
- **Configurable confidence threshold** — Filter weak detections
- **Input validation** — File type checking, parameter validation
- **Auto-generated API docs** — Interactive Swagger UI at `/docs`
- **80 detectable classes** — Person, car, dog, cat, chair, phone, and 74 more

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — verify API and model status |
| `GET` | `/classes` | List all 80 detectable object classes |
| `POST` | `/detect` | Upload image → get JSON detection results |
| `POST` | `/detect/image` | Upload image → get annotated image back |
| `GET` | `/docs` | Interactive Swagger API documentation |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
python main.py
```
The server starts at `http://localhost:8000`

### 3. Test with curl
```bash
# Get detection results as JSON
curl -X POST "http://localhost:8000/detect?confidence=0.5" \
  -F "file=@your_photo.jpg"

# Get annotated image with bounding boxes
curl -X POST "http://localhost:8000/detect/image?confidence=0.5" \
  -F "file=@your_photo.jpg" --output result.jpg
```

### 4. Or use the interactive docs
Open `http://localhost:8000/docs` in your browser — upload images directly from the Swagger UI.

## Sample Response

```json
{
  "image_info": {
    "filename": "street.jpg",
    "width": 1920,
    "height": 1080,
    "channels": 3
  },
  "detection_config": {
    "model": "YOLOv8n",
    "confidence_threshold": 0.5
  },
  "summary": {
    "total_objects_detected": 3,
    "objects_per_class": {
      "person": 2,
      "car": 1
    }
  },
  "detections": [
    {
      "class": "person",
      "confidence": 0.934,
      "bounding_box": {
        "x1": 100, "y1": 200, "x2": 250, "y2": 600,
        "width": 150, "height": 400
      }
    }
  ],
  "annotated_image_url": "/static/results/a3f2c1d8.jpg"
}
```

## Architecture

```
Client (Browser / curl / Flutter app)
    │
    │  POST /detect  (image upload)
    ▼
┌──────────────────────────────┐
│  FastAPI Server              │
│  ┌────────────────────────┐  │
│  │  Input Validation      │  │  File type check, confidence range
│  ├────────────────────────┤  │
│  │  Image Decoding        │  │  Bytes → NumPy array (OpenCV)
│  ├────────────────────────┤  │
│  │  YOLOv8 Inference      │  │  Pre-loaded model, single forward pass
│  ├────────────────────────┤  │
│  │  Result Parsing        │  │  Bounding boxes, classes, confidence
│  ├────────────────────────┤  │
│  │  Response Formatting   │  │  Structured JSON + annotated image
│  └────────────────────────┘  │
└──────────────────────────────┘
    │
    ▼
  JSON Response + Annotated Image
```

## Author

**Deva Nandan H** — B.Tech CSE, CUSAT  
[GitHub](https://github.com/Devan433) | [LinkedIn](https://linkedin.com/in/deva-nandan-h)
