"""
Gera versões escuras das imagens de validação.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np


SRC = Path(
    "dataset/exports/epi-v1/valid"
)

DEST = Path(
    "dataset/exports/epi-v1-dark/valid"
)


(DEST / "images").mkdir(
    parents=True,
    exist_ok=True,
)

(DEST / "labels").mkdir(
    parents=True,
    exist_ok=True,
)


for lbl in (
    SRC / "labels"
).glob("*.txt"):
    shutil.copy(
        lbl,
        DEST / "labels" / lbl.name,
    )


gamma = 2.2

table = np.array(
    [
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]
).astype(np.uint8)


for img_path in (
    SRC / "images"
).glob("*.jpg"):

    img = cv2.imread(
        str(img_path)
    )

    dark = cv2.LUT(
        img,
        table,
    )

    cv2.imwrite(
        str(
            DEST
            / "images"
            / img_path.name
        ),
        dark,
    )


count = len(
    list(
        (DEST / "images")
        .glob("*.jpg")
    )
)

print(
    f"Geradas {count} "
    "imagens escurecidas"
)
