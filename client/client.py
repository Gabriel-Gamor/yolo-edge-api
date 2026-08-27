import base64
import os
import json
import time
from pathlib import Path

import httpx


API_URL = os.getenv("API_URL", "http://localhost:8000")

IMAGES_DIR = Path("/client/images")
OUTPUT_DIR = Path("/client/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def wait_for_api(max_retries: int = 10, delay: float = 3.0):
    for attempt in range(max_retries):
        try:
            response = httpx.get(
                f"{API_URL}/health",
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()

                print(
                    f"[OK] API disponível | "
                    f"modelo: {data['model_name']}"
                )

                return True

        except httpx.ConnectError:
            pass

        print(
            f"[...] Aguardando API "
            f"({attempt + 1}/{max_retries})..."
        )

        time.sleep(delay)

    raise RuntimeError(
        "API não ficou disponível a tempo."
    )


def run_single_inference(
    image_path: Path,
    confidence: float = 0.25
):
    print(
        f"\n--- Inferência: "
        f"{image_path.name} ---"
    )

    payload = {
        "image_base64": encode_image(image_path),
        "confidence": confidence,
        "model_name": "yolov8n.pt",
    }

    response = httpx.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    print(
        f"Tempo de inferência: "
        f"{data['inference_ms']} ms"
    )

    print(
        f"Resolução: "
        f"{data['image_width']}x"
        f"{data['image_height']} px"
    )

    print(
        f"Detecções: "
        f"{len(data['detections'])}"
    )

    for det in data["detections"]:
        print(
            f" - {det['label']} "
            f"conf={det['confidence']:.2f}"
        )

    out_file = (
        OUTPUT_DIR /
        f"{image_path.stem}_result.json"
    )

    out_file.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )
    )

    print(
        f"Resultado salvo em: "
        f"{out_file}"
    )


if __name__ == "__main__":

    wait_for_api()

    images = (
        sorted(IMAGES_DIR.glob("*.jpg")) +
        sorted(IMAGES_DIR.glob("*.png"))
    )

    if not images:
        print(
            "[AVISO] Nenhuma imagem encontrada "
            "em /client/images/"
        )
    else:
        run_single_inference(images[0])
