#!/usr/bin/env python3
"""
stream/v3_optimized.py — Pipeline otimizado para tempo real no
Raspberry Pi 5.

Combina:
- threading
- frame skip
- resolução adaptativa
- OSD
- pré-processamento reutilizável

Saída:
- vídeo anotado via cv2.imshow()
- ou arquivo .avi em modo headless

Execução:
python3 stream/v3_optimized.py --device 0 --infer-every 3
"""

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ---------------------------------------------------------
# Permite importar módulos da raiz do projeto
# ---------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).parent.parent),
)


# ---------------------------------------------------------
# Pré-processamento — Aula 5
# ---------------------------------------------------------

from preprocessing.preprocessor import (
    PreprocessConfig,
    Preprocessor,
)


# ---------------------------------------------------------
# Compatibilidade do torch.load
# ---------------------------------------------------------

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return _orig_torch_load(
        *args,
        **kwargs,
    )


torch.load = _patched_torch_load


# ---------------------------------------------------------
# Câmera otimizada
# ---------------------------------------------------------

class OptimizedCamera:
    """
    Câmera com MJPEG nativo, FPS fixo e buffer de 1 frame.

    O formato MJPEG evita conversões desnecessárias
    na CPU e o buffer mantém sempre o frame mais recente.
    """

    def __init__(
        self,
        device: int,
        width: int,
        height: int,
        fps: int = 30,
        use_mjpeg: bool = True,
    ):
        self._cmd = [
            "rpicam-vid",
            "-t",
            "0",
            "-n",
            "--codec",
            "mjpeg",
            "--camera",
            str(device),
            "--width",
            str(width),
            "--height",
            str(height),
            "--framerate",
            str(fps),
            "-o",
            "-",
        ]

        self._proc = None
        self._raw = b""

        print(
            f"[OptimizedCamera] "
            f"Resolução solicitada: "
            f"{width}x{height} @ {fps} FPS"
        )

        self._buf = queue.Queue(
            maxsize=1
        )

        self._running = (
            threading.Event()
        )

        self._running.set()

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="CamThread",
        )

        self.frames_in = 0
        self.frames_out = 0

    def start(self):
        self._proc = subprocess.Popen(
            self._cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        self._thread.start()

        return self

    def _loop(self):
        while self._running.is_set():

            chunk = (
                self._proc.stdout.read(
                    4096
                )
            )

            if not chunk:
                break

            self._raw += chunk

            end = self._raw.rfind(
                b"\xff\xd9"
            )

            if end == -1:
                continue

            start = self._raw.rfind(
                b"\xff\xd8",
                0,
                end,
            )

            if start == -1:
                continue

            jpg = self._raw[
                start:end + 2
            ]

            self._raw = self._raw[
                end + 2:
            ]

            frame = cv2.imdecode(
                np.frombuffer(
                    jpg,
                    dtype=np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                continue

            self.frames_in += 1

            if self._buf.full():
                try:
                    self._buf.get_nowait()

                except queue.Empty:
                    pass

            self._buf.put(frame)

    def read(
        self,
        timeout=1.0,
    ):
        try:
            frame = self._buf.get(
                timeout=timeout
            )

            self.frames_out += 1

            return frame

        except queue.Empty:
            return None

    def stop(self):
        self._running.clear()

        if self._proc:
            self._proc.terminate()

            try:
                self._proc.wait(
                    timeout=2.0
                )

            except subprocess.TimeoutExpired:
                self._proc.kill()

        self._thread.join(
            timeout=2.0
        )


# ---------------------------------------------------------
# Detector YOLO
# ---------------------------------------------------------

class RealtimeDetector:
    """
    Detector YOLO com frame skip e OSD.

    Mantém o último resultado para exibição
    nos frames intermediários.

    O pré-processamento é realizado pelo
    módulo reutilizável Preprocessor.
    """

    def __init__(
        self,
        model_path: str,
        conf: float,
        infer_every: int,
        infer_size: int,
    ):
        print(
            f"[RealtimeDetector] "
            f"Modelo: {model_path}"
        )

        print(
            f"[RealtimeDetector] "
            f"Inferência a cada "
            f"{infer_every} frames"
        )

        print(
            f"[RealtimeDetector] "
            f"Tamanho de inferência: "
            f"{infer_size}px"
        )

        self.model = YOLO(
            model_path
        )

        self.conf = conf

        self.infer_every = (
            infer_every
        )

        self.infer_size = (
            infer_size
        )

        # ---------------------------------------------
        # Integração do Preprocessor — Aula 5
        # ---------------------------------------------

        self.preprocessor = (
            Preprocessor(
                PreprocessConfig(
                    infer_size=infer_size
                )
            )
        )

        self._frame_idx = 0

        self._last_boxes = []

        self._last_infer_ms = 0.0

        # FPS calculado sobre janela
        # deslizante de 30 frames
        self._fps_window = deque(
            maxlen=30
        )

        self._t_last = (
            time.perf_counter()
        )

    def process(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Processa um frame.

        - Executa YOLO a cada infer_every frames.
        - Nos demais frames reutiliza as últimas boxes.
        - Usa Preprocessor antes da inferência.
        - Converte as bounding boxes de volta
          para as coordenadas da imagem original.
        """

        self._frame_idx += 1

        # ---------------------------------------------
        # Atualização do FPS
        # ---------------------------------------------

        now = time.perf_counter()

        dt = (
            now - self._t_last
        )

        self._t_last = now

        self._fps_window.append(
            dt
        )

        # ---------------------------------------------
        # Inferência a cada N frames
        # ---------------------------------------------

        if (
            self._frame_idx
            % self.infer_every
            == 0
        ):

            # -----------------------------------------
            # Pré-processamento — Aula 5
            # -----------------------------------------

            preproc_result = (
                self.preprocessor.process(
                    frame
                )
            )

            # -----------------------------------------
            # Inferência YOLO
            # -----------------------------------------

            t0 = time.perf_counter()

            results = self.model(
                preproc_result.frame,
                conf=self.conf,
                verbose=False,
            )

            self._last_infer_ms = (
                time.perf_counter()
                - t0
            ) * 1000

            self._last_boxes = []

            # -----------------------------------------
            # Ajusta bounding boxes para
            # coordenadas da imagem original
            # -----------------------------------------

            for r in results:

                for box in r.boxes:

                    bbox_lb = (
                        box.xyxy[0]
                        .numpy()
                        .reshape(
                            1,
                            4,
                        )
                    )

                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    ) = (
                        self.preprocessor
                        .adjust_boxes(
                            bbox_lb,
                            preproc_result,
                        )[0]
                    )

                    label = (
                        self.model.names[
                            int(
                                box.cls[0]
                            )
                        ]
                    )

                    conf = float(
                        box.conf[0]
                    )

                    self._last_boxes.append(
                        (
                            label,
                            conf,
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        )
                    )

        # ---------------------------------------------
        # Desenha bounding boxes
        # ---------------------------------------------

        output = frame.copy()

        for (
            label,
            conf,
            x1,
            y1,
            x2,
            y2,
        ) in self._last_boxes:

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            caption = (
                f"{label} "
                f"{conf:.0%}"
            )

            (tw, th), _ = (
                cv2.getTextSize(
                    caption,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    1,
                )
            )

            cv2.rectangle(
                output,
                (
                    x1,
                    y1 - th - 8,
                ),
                (
                    x1 + tw + 4,
                    y1,
                ),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                output,
                caption,
                (
                    x1 + 2,
                    y1 - 4,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
            )

        # ---------------------------------------------
        # OSD
        # ---------------------------------------------

        if self._fps_window:
            fps_display = (
                len(
                    self._fps_window
                )
                /
                sum(
                    self._fps_window
                )
            )

        else:
            fps_display = 0

        is_infer_frame = (
            self._frame_idx
            % self.infer_every
            == 0
        )

        osd_lines = [
            (
                f"FPS: "
                f"{fps_display:.1f}"
            ),
            (
                f"Infer: "
                f"{self._last_infer_ms:.0f}ms"
            ),
            (
                f"Det: "
                f"{len(self._last_boxes)}"
            ),
            (
                f"Frame: "
                f"{self._frame_idx}"
            ),
        ]

        for i, line in enumerate(
            osd_lines
        ):
            y = 28 + i * 26

            color = (
                (0, 255, 255)
                if is_infer_frame
                else (200, 200, 200)
            )

            cv2.putText(
                output,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return output


# ---------------------------------------------------------
# Argumentos da linha de comando
# ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--device",
        type=int,
        default=0,
    )

    p.add_argument(
        "--width",
        type=int,
        default=640,
    )

    p.add_argument(
        "--height",
        type=int,
        default=480,
    )

    p.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    p.add_argument(
        "--model",
        type=str,
        default="models/yolov8n.pt",
    )

    p.add_argument(
        "--conf",
        type=float,
        default=0.4,
    )

    p.add_argument(
        "--infer-every",
        type=int,
        default=3,
        help=(
            "Executa YOLO a cada N frames "
            "(padrão: 3)"
        ),
    )

    p.add_argument(
        "--infer-size",
        type=int,
        default=320,
        help=(
            "Resolução de inferência em px "
            "(padrão: 320)"
        ),
    )

    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Salva o stream anotado "
            "em arquivo .avi (opcional)"
        ),
    )

    p.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "Desativa cv2.imshow() "
            "(modo headless/SSH)"
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    args = parse_args()

    camera = OptimizedCamera(
        args.device,
        args.width,
        args.height,
        args.fps,
    )

    detector = RealtimeDetector(
        args.model,
        args.conf,
        args.infer_every,
        args.infer_size,
    )

    writer = None

    if args.output:
        fourcc = (
            cv2.VideoWriter_fourcc(
                *"XVID"
            )
        )

        writer = cv2.VideoWriter(
            args.output,
            fourcc,
            args.fps,
            (
                args.width,
                args.height,
            ),
        )

        print(
            f"[INFO] Gravando saída em: "
            f"{args.output}"
        )

    camera.start()

    time.sleep(0.5)

    print(
        "[INFO] Stream iniciado. "
        "Pressione Ctrl+C para encerrar."
    )

    if not args.no_display:
        print(
            "[INFO] Pressione 'q' "
            "na janela para encerrar."
        )

    try:

        while True:

            frame = camera.read(
                timeout=2.0
            )

            if frame is None:
                print(
                    "[AVISO] "
                    "Timeout na leitura."
                )

                continue

            annotated = (
                detector.process(
                    frame
                )
            )

            if writer:
                writer.write(
                    annotated
                )

            if not args.no_display:

                cv2.imshow(
                    "YOLO — Tempo Real "
                    "(pressione q para sair)",
                    annotated,
                )

                if (
                    cv2.waitKey(1)
                    & 0xFF
                    == ord("q")
                ):
                    break

    except KeyboardInterrupt:
        print(
            "\n[INFO] "
            "Encerrado pelo usuário."
        )

    finally:

        camera.stop()

        if writer:
            writer.release()

        cv2.destroyAllWindows()

        print(
            f"[INFO] Frames processados: "
            f"{detector._frame_idx}"
        )


if __name__ == "__main__":
    main()
