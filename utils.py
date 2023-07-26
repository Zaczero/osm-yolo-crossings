import os
import time
from contextlib import contextmanager
from math import cos, pi, radians
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.io import imsave

from config import IMAGES_DIR, SAVE_IMG, USER_AGENT


@contextmanager
def print_run_time(message: str | list) -> Generator[None, None, None]:
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # support message by reference
        if isinstance(message, list):
            message = message[0]

        print(f'[⏱️] {message} took {elapsed_time:.3f}s')


def http_headers() -> dict:
    return {
        'User-Agent': USER_AGENT,
    }


def save_image(image: np.ndarray, name: str = 'UNTITLED', *, final_path: bool = False, force: bool = False) -> Path | None:
    if not SAVE_IMG and not force:
        return None

    if image.dtype in ('float32', 'float64', 'bool'):
        image = img_as_ubyte(image)

    if final_path:
        image_path = Path(f'{name}.png')
    else:
        image_path = IMAGES_DIR / f'{name}.png'

    if not image_path.parent.is_dir():
        os.makedirs(image_path.parent, exist_ok=True)

    imsave(image_path, image, check_contrast=False)
    return image_path


def random_color() -> np.ndarray:
    color = np.random.rand(3)
    color = np.maximum(color, 0.2)
    return color


EARTH_RADIUS = 6371000
CIRCUMFERENCE = 2 * pi * EARTH_RADIUS


def meters_to_lat(meters: float) -> float:
    return meters / (CIRCUMFERENCE / 360)


def meters_to_lon(meters: float, lat: float) -> float:
    return meters / ((CIRCUMFERENCE / 360) * cos(radians(lat)))


def draw_predictions(image: np.ndarray, y_pred: dict, i: int) -> np.ndarray:
    boxes = y_pred['boxes'][i]
    confidence = y_pred['confidence'][i]
    classes = y_pred['classes'][i]
    num_detections = y_pred['num_detections'][i]

    image = img_as_ubyte(image)

    for box, confidence, class_id in zip(boxes[:num_detections], confidence[:num_detections], classes[:num_detections]):
        if confidence < 0.5:
            continue

        x, y, w, h = box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        text = f'{confidence:.2f}'
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image
