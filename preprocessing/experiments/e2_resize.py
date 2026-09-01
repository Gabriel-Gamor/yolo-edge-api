"""
Experimento E2:
resize ingênuo vs letterbox.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")

from preprocessing.utils.evaluate import (
    evaluate_pipeline,
)

from preprocessing.utils.letterbox import (
    adjust_bboxes,
    letterbox,
)


TARGET = 416


def preproc_naive_resize(
    frame: np.ndarray,
) -> np.ndarray:
    return cv2.resize(
        frame,
        (TARGET, TARGET),
    )


def preproc_letterbox(
    frame: np.ndarray,
) -> np.ndarray:
    frame_lb, _, _ = letterbox(
        frame,
        target_size=TARGET,
    )

    return frame_lb


def demo_bbox_adjustment():
    img_path = sorted(
        Path(
            "dataset/exports/epi-v1/"
            "valid/images"
        ).glob("*.jpg")
    )[0]

    frame = cv2.imread(
        str(img_path)
    )

    h_orig, w_orig = frame.shape[:2]

    frame_lb, scale, (
        pad_w,
        pad_h,
    ) = letterbox(
        frame,
        target_size=TARGET,
    )

    bbox_lb = np.array(
        [[60, 90, 200, 310]],
        dtype=float,
    )

    bbox_orig = adjust_bboxes(
        bbox_lb,
        scale,
        pad_w,
        pad_h,
    )

    print(
        f"Frame original: "
        f"{w_orig}x{h_orig}"
    )

    print(
        f"Frame letterboxed: "
        f"{TARGET}x{TARGET} "
        f"scale={scale:.4f} "
        f"pad_w={pad_w} "
        f"pad_h={pad_h}"
    )

    print(
        "Bbox no espaço letterboxed:",
        bbox_lb[0]
        .astype(int)
        .tolist(),
    )

    print(
        "Bbox ajustada ao original:",
        bbox_orig[0]
        .astype(int)
        .tolist(),
    )

    cv2.rectangle(
        frame_lb,
        tuple(
            bbox_lb[0, :2]
            .astype(int)
        ),
        tuple(
            bbox_lb[0, 2:]
            .astype(int)
        ),
        (0, 255, 0),
        2,
    )

    cv2.rectangle(
        frame,
        tuple(
            bbox_orig[0, :2]
            .astype(int)
        ),
        tuple(
            bbox_orig[0, 2:]
            .astype(int)
        ),
        (0, 255, 0),
        2,
    )

    cv2.imwrite(
        "preprocessing/outputs/"
        "e2_bbox_letterboxed.jpg",
        frame_lb,
    )

    cv2.imwrite(
        "preprocessing/outputs/"
        "e2_bbox_original.jpg",
        frame,
    )


if __name__ == "__main__":
    print("=" * 65)
    print(
        " E2 — Resize Ingênuo "
        "vs Letterbox"
    )
    print("=" * 65)

    results = []

    results.append(
        evaluate_pipeline(
            None,
            "E2-baseline",
        )
    )

    results.append(
        evaluate_pipeline(
            preproc_naive_resize,
            "E2-A: resize ingênuo",
        )
    )

    results.append(
        evaluate_pipeline(
            preproc_letterbox,
            "E2-B: letterbox",
        )
    )

    print(
        "\n--- Demonstração de "
        "ajuste de coordenadas ---"
    )

    demo_bbox_adjustment()

    print("\n--- Resumo E2 ---")

    baseline = results[0]["map50"]

    for result in results[1:]:
        print(
            f" {result['label']:30s} "
            f"mAP@0.5="
            f"{result['map50']:.4f} "
            f"delta="
            f"{result['map50'] - baseline:+.4f}"
        )
