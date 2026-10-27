"""YOLOX (OpenCV Zoo, Apache-2.0) on the OpenCV 5 DNN engine.

Pre/post-processing follows opencv_zoo's yolox.py (verified on opencv-python 5.0.0.93, NEW and CLASSIC
engines give identical detections): RGB, letterbox to 640 padded with 114, raw 0-255 input,
grid/stride decode, score = objectness * class score, class-aware NMS via cv2.dnn.NMSBoxes.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parents[2] / "data" / "models" / "object_detection_yolox_2022nov.onnx"
MODEL_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "object_detection_yolox/object_detection_yolox_2022nov.onnx")
INPUT = 640
STRIDES = (8, 16, 32)
COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


@dataclass
class Detection:
    cls: str
    score: float
    box: tuple[float, float, float, float]   # x, y, w, h in original-image pixels


def _grids():
    grids, strides = [], []
    for s in STRIDES:
        n = INPUT // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


GRIDS, EXP_STRIDES = _grids()


class YoloxDetector:
    def __init__(self, model: Path = MODEL, engine: int = cv2.dnn.ENGINE_NEW,
                 conf: float = 0.25, nms: float = 0.5, classes: tuple[str, ...] | None = None):
        if not Path(model).exists():
            raise FileNotFoundError(f"{model} missing; download it from {MODEL_URL}")
        self.net = cv2.dnn.readNetFromONNX(str(model), engine)
        self.out_names = self.net.getUnconnectedOutLayersNames()
        self.conf, self.nms = conf, nms
        self.keep = None if classes is None else {COCO.index(c) for c in classes}

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        blob, ratio = self._letterbox(bgr)
        self.net.setInput(blob)
        out = self.net.forward(self.out_names)[0]
        return self._post(out, ratio)

    @staticmethod
    def _letterbox(bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ratio = min(INPUT / rgb.shape[0], INPUT / rgb.shape[1])
        w, h = int(rgb.shape[1] * ratio), int(rgb.shape[0] * ratio)
        padded = np.full((INPUT, INPUT, 3), 114.0, np.float32)
        padded[:h, :w] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(padded.transpose(2, 0, 1)[None]), ratio

    def _post(self, out: np.ndarray, ratio: float) -> list[Detection]:
        d = out.reshape(-1, 85).copy()
        d[:, :2] = (d[:, :2] + GRIDS) * EXP_STRIDES
        d[:, 2:4] = np.exp(d[:, 2:4]) * EXP_STRIDES
        scores = d[:, 4:5] * d[:, 5:]
        cls, score = scores.argmax(1), scores.max(1)
        xywh = np.stack([d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2, d[:, 2], d[:, 3]], 1) / ratio
        m = score >= self.conf
        if self.keep is not None:
            m &= np.isin(cls, list(self.keep))
        xywh, score, cls = xywh[m], score[m], cls[m]
        if len(score) == 0:
            return []
        shifted = xywh.copy()
        shifted[:, :2] += cls[:, None] * 4096.0     # class-aware NMS: offset boxes per class
        keep = np.array(cv2.dnn.NMSBoxes(shifted.tolist(), score.tolist(), self.conf, self.nms)).reshape(-1)
        dets = [Detection(COCO[int(cls[i])], float(score[i]), tuple(float(v) for v in xywh[i])) for i in keep]
        return sorted(dets, key=lambda x: -x.score)
