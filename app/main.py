import base64
import io
import time

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from model import get_default_model_name, load_model
from PIL import Image
from schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Detection,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(
    title="YOLO Inference API",
    description="API REST para inferencia com YOLOv8 no Raspberry Pi 5",
    version="1.0.0",
)


_metrics = {
    "total": 0,
    "success": 0,
    "total_ms": 0.0
}


def _decode_image(image_base64: str) -> np.ndarray:
    raw = base64.b64decode(image_base64)

    img = Image.open(
        io.BytesIO(raw)
    ).convert("RGB")

    return np.array(img)


def _load_image_from_request(
    request: PredictRequest
) -> np.ndarray:

    if not request.image_base64 and not request.image_url:
        raise HTTPException(
            status_code=422,
            detail="Forneca image_base64 ou image_url."
        )

    if request.image_base64:
        return _decode_image(
            request.image_base64
        )

    response = httpx.get(
        request.image_url,
        timeout=15.0,
        follow_redirects=True
    )

    response.raise_for_status()

    img = Image.open(
        io.BytesIO(response.content)
    ).convert("RGB")

    return np.array(img)


def _run_inference(
    image_np: np.ndarray,
    model_name: str,
    confidence: float
) -> PredictResponse:

    model = load_model(model_name)

    t0 = time.perf_counter()

    results = model(
        image_np,
        conf=confidence,
        verbose=False
    )

    elapsed_ms = (
        time.perf_counter() - t0
    ) * 1000

    detections = []

    for result in results:
        for box in result.boxes:

            coords = box.xyxy[0].tolist()

            class_id = int(
                box.cls[0].item()
            )

            confidence_value = float(
                box.conf[0].item()
            )

            detections.append(
                Detection(
                    label=model.names[class_id],
                    confidence=round(
                        confidence_value,
                        4
                    ),
                    bbox=[
                        round(float(c), 2)
                        for c in coords
                    ]
                )
            )

    h, w = image_np.shape[:2]

    return PredictResponse(
        detections=detections,
        inference_ms=round(elapsed_ms, 2),
        model_used=model_name,
        image_width=w,
        image_height=h
    )


@app.get(
    "/health",
    response_model=HealthResponse
)
async def health_check():

    model_name = get_default_model_name()

    try:
        load_model(model_name)
        loaded = True

    except Exception:
        loaded = False

    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_name=model_name
    )


@app.post(
    "/predict",
    response_model=PredictResponse
)
def predict(
    request: PredictRequest
):

    _metrics["total"] += 1

    try:
        image = _load_image_from_request(
            request
        )

        result = _run_inference(
            image,
            request.model_name,
            request.confidence
        )

        _metrics["success"] += 1
        _metrics["total_ms"] += (
            result.inference_ms
        )

        return result

    except HTTPException:
        raise

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post(
    "/predict/image",
    responses={
        200: {
            "content": {
                "image/jpeg": {}
            }
        }
    }
)
def predict_image(
    request: PredictRequest
):

    _metrics["total"] += 1

    try:
        img_rgb = _load_image_from_request(
            request
        )

        model = load_model(
            request.model_name
        )

        t0 = time.perf_counter()

        results = model(
            img_rgb,
            conf=request.confidence,
            verbose=False
        )

        elapsed_ms = (
            time.perf_counter() - t0
        ) * 1000

        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms

        annotated_array = (
            results[0].plot()
        )

        annotated_pil = Image.fromarray(
            annotated_array
        )

        buffer = io.BytesIO()

        annotated_pil.save(
            buffer,
            format="JPEG",
            quality=95
        )

        return Response(
            content=buffer.getvalue(),
            media_type="image/jpeg"
        )

    except HTTPException:
        raise

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse
)
def predict_batch(
    request: BatchPredictRequest
):

    t_total = time.perf_counter()

    results = []

    for img_b64 in request.images_base64:

        image = _decode_image(
            img_b64
        )

        results.append(
            _run_inference(
                image,
                request.model_name,
                request.confidence
            )
        )

    total_ms = (
        time.perf_counter() - t_total
    ) * 1000

    return BatchPredictResponse(
        results=results,
        total_inference_ms=round(
            total_ms,
            2
        )
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse
)
async def get_metrics():

    avg = (
        _metrics["total_ms"]
        / _metrics["success"]
        if _metrics["success"] > 0
        else 0.0
    )

    return MetricsResponse(
        total_requests=_metrics["total"],
        successful_requests=_metrics["success"],
        avg_inference_ms=round(avg, 2)
    )
