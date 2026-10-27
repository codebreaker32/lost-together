"""YOLOX (OpenCV Zoo, 2022nov fp32 ONNX) check on OpenCV 5: NEW vs CLASSIC DNN engine.

Model:  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx
        (Apache-2.0, see https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox)
Pre/post-processing follows opencv_zoo's yolox.py / demo.py (RGB, letterbox pad 114, raw 0-255 input,
grid/stride decode, score = obj * cls).

Usage: python -I yolox_check.py MODEL.onnx IMAGE [IMAGE ...]
"""
import sys
import time

import cv2
import numpy as np

COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
INPUT = 640
STRIDES = (8, 16, 32)
CONF_T, NMS_T = 0.35, 0.5
RUNS = 10


def make_grids():
    grids, strides = [], []
    for s in STRIDES:
        n = INPUT // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


GRIDS, EXP_STRIDES = make_grids()


def letterbox(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ratio = min(INPUT / rgb.shape[0], INPUT / rgb.shape[1])
    w, h = int(rgb.shape[1] * ratio), int(rgb.shape[0] * ratio)
    padded = np.full((INPUT, INPUT, 3), 114.0, np.float32)
    padded[:h, :w] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None])  # 1x3x640x640, 0-255
    return blob, ratio


def postprocess(out, ratio):
    dets = out.reshape(-1, 85).copy()
    dets[:, :2] = (dets[:, :2] + GRIDS) * EXP_STRIDES
    dets[:, 2:4] = np.exp(dets[:, 2:4]) * EXP_STRIDES
    scores_all = dets[:, 4:5] * dets[:, 5:]
    cls = scores_all.argmax(1)
    score = scores_all.max(1)
    xywh = np.stack([dets[:, 0] - dets[:, 2] / 2, dets[:, 1] - dets[:, 3] / 2, dets[:, 2], dets[:, 3]], 1) / ratio
    m = score >= CONF_T
    xywh, score, cls = xywh[m], score[m], cls[m]
    if len(score) == 0:
        return []
    # class-aware NMS via cv2.dnn.NMSBoxes: offset boxes per class so different classes never overlap
    off = (cls[:, None] * 4096.0).astype(np.float32)
    shifted = xywh.copy()
    shifted[:, :2] += off
    keep = cv2.dnn.NMSBoxes(shifted.tolist(), score.tolist(), CONF_T, NMS_T)
    keep = np.array(keep).reshape(-1)
    res = [(COCO[int(cls[i])], float(score[i]), [round(float(v), 1) for v in xywh[i]]) for i in keep]
    return sorted(res, key=lambda r: -r[1])


def run_engine(model, engine_name, engine, images):
    print(f"\n=== engine {engine_name} (cv2.dnn.{engine_name} = {engine}) ===")
    try:
        net = cv2.dnn.readNetFromONNX(model, engine=engine)
    except Exception as e:  # noqa: BLE001
        print(f"LOAD FAILED: {e!r}")
        return None
    out_names = net.getUnconnectedOutLayersNames()
    results = {}
    for path in images:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        blob, ratio = letterbox(img)
        try:
            net.setInput(blob)
            out = net.forward(out_names)[0]  # warm-up / first run
            times = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                net.setInput(blob)
                out = net.forward(out_names)[0]
                times.append((time.perf_counter() - t0) * 1000)
        except Exception as e:  # noqa: BLE001
            print(f"FORWARD FAILED on {path}: {e!r}")
            return None
        dets = postprocess(out, ratio)
        med = float(np.median(times))
        print(f"{path.split('/')[-1]}: out shape {tuple(out.shape)}, median {med:.1f} ms over {RUNS} runs "
              f"(min {min(times):.1f}, max {max(times):.1f})")
        for name, s, box in dets[:10]:
            print(f"   {name:<12s} {s:.3f}  xywh={box}")
        results[path] = (med, dets, out)
    return results


def main():
    model, images = sys.argv[1], sys.argv[2:]
    print("cv2", cv2.__version__, "| threads", cv2.getNumThreads())
    print("ENGINE constants:", {a: getattr(cv2.dnn, a) for a in dir(cv2.dnn) if a.startswith("ENGINE_")})
    new = run_engine(model, "ENGINE_NEW", cv2.dnn.ENGINE_NEW, images)
    classic = run_engine(model, "ENGINE_CLASSIC", cv2.dnn.ENGINE_CLASSIC, images)
    if new and classic:
        print("\n=== NEW vs CLASSIC raw-output agreement ===")
        for p in images:
            d = np.abs(new[p][2] - classic[p][2]).max()
            print(f"{p.split('/')[-1]}: max |diff| = {d:.2e}; median ms new={new[p][0]:.1f} classic={classic[p][0]:.1f}")


if __name__ == "__main__":
    main()
