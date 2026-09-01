"""
Gera comparativo visual BGR vs RGB.
"""

from pathlib import Path

import cv2


images = sorted(
    Path(
        "dataset/exports/epi-v1/valid/images"
    ).glob("*.jpg")
)

img_path = images[0]

frame = cv2.imread(str(img_path))

rgb = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2RGB,
)

# Imagem original OpenCV
cv2.imwrite(
    "preprocessing/outputs/e1_bgr_original.jpg",
    frame,
)

# Converte RGB novamente para BGR somente
# para o cv2.imwrite salvar as cores corretamente.
cv2.imwrite(
    "preprocessing/outputs/e1_rgb_correto.jpg",
    cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    ),
)

print(
    "Imagens salvas em "
    "preprocessing/outputs/"
)
