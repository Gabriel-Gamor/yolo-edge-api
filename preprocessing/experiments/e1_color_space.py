"""
Experimento E1:
impacto da conversão de espaço de cor BGR -> RGB.
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")

from preprocessing.utils.evaluate import evaluate_pipeline


def preproc_bgr_raw(
    frame: np.ndarray,
) -> np.ndarray:
    """Não converte BGR para RGB."""
    return frame


def preproc_rgb_correct(
    frame: np.ndarray,
) -> np.ndarray:
    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )


def preproc_rgb_flip(
    frame: np.ndarray,
) -> np.ndarray:
    return frame[:, :, ::-1]


if __name__ == "__main__":
    print("=" * 65)
    print(
        " E1 — Impacto da Conversão "
        "de Espaço de Cor"
    )
    print("=" * 65)

    results = []

    results.append(
        evaluate_pipeline(
            None,
            "E1-baseline (Ultralytics padrão)",
        )
    )

    results.append(
        evaluate_pipeline(
            preproc_bgr_raw,
            "E1-A: BGR sem conversão",
        )
    )

    results.append(
        evaluate_pipeline(
            preproc_rgb_correct,
            "E1-B: BGR→RGB (cvtColor)",
        )
    )

    results.append(
        evaluate_pipeline(
            preproc_rgb_flip,
            "E1-C: BGR→RGB (NumPy flip)",
        )
    )

    print("\n--- Resumo E1 ---")

    baseline_map = results[0]["map50"]

    for r in results[1:]:
        delta = r["map50"] - baseline_map
        sinal = "+" if delta >= 0 else ""

        print(
            f" {r['label']:35s} "
            f"mAP@0.5={r['map50']:.4f} "
            f"delta={sinal}{delta:.4f}"
        )
