from pathlib import Path

import cv2
import pytest

from lost_together.vision.detector import MODEL, YoloxDetector

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="YOLOX model not downloaded")


def test_yolox_finds_people_on_reference_image():
    """Reference: opencv/opencv samples/data/basketball1.png gave person 0.926 and 0.846 on both engines."""
    img = cv2.imread(str(Path(__file__).with_name("basketball1.png")))
    dets = YoloxDetector(classes=("person",)).detect(img)
    assert len(dets) >= 2 and dets[0].score > 0.9
