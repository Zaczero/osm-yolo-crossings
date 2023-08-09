import os
import time
from contextlib import contextmanager
from math import atan2, cos, dist, pi, radians, sin, sqrt
from pathlib import Path
from typing import Generator, Sequence

import cv2
import numpy as np
import tensorflow as tf
from numba import njit
from skimage import img_as_ubyte
from skimage.io import imsave

from config import IMAGES_DIR, SAVE_IMG, USER_AGENT, YOLO_CONFIDENCE
from latlon import LatLon


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


EARTH_RADIUS = 6371000
CIRCUMFERENCE = 2 * pi * EARTH_RADIUS


@njit(fastmath=True)
def meters_to_lat(meters: float) -> float:
    return meters / (CIRCUMFERENCE / 360)


@njit(fastmath=True)
def meters_to_lon(meters: float, lat: float) -> float:
    return meters / ((CIRCUMFERENCE / 360) * cos(radians(lat)))


@njit(fastmath=True)
def lat_to_meters(lat: float) -> float:
    return lat * (CIRCUMFERENCE / 360)


@njit(fastmath=True)
def lon_to_meters(lon: float, lat: float) -> float:
    return lon * ((CIRCUMFERENCE / 360) * cos(radians(lat)))


@njit(fastmath=True)
def radians_tuple(p: tuple[float, float]) -> tuple[float, float]:
    return (radians(p[0]), radians(p[1]))


@njit(fastmath=True)
def haversine_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    p1_lat, p1_lon = radians_tuple(p1)
    p2_lat, p2_lon = radians_tuple(p2)

    dlat = p2_lat - p1_lat
    dlon = p2_lon - p1_lon

    a = sin(dlat / 2)**2 + cos(p1_lat) * cos(p2_lat) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # distance in meters
    return c * EARTH_RADIUS


def draw_predictions(image: np.ndarray, y_pred: dict, i: int) -> np.ndarray:
    boxes = y_pred['boxes'][i]
    confidence = y_pred['confidence'][i]
    classes = y_pred['classes'][i]
    num_detections = y_pred['num_detections'][i]

    image = img_as_ubyte(image)

    for box, confidence, class_id in zip(boxes[:num_detections], confidence[:num_detections], classes[:num_detections]):
        if confidence < YOLO_CONFIDENCE:
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


def index_box_centered(boxes: Sequence[tuple], resolution: int) -> int:
    assert boxes
    center_x = resolution / 2
    center_y = resolution / 2

    best_i = None
    best_dist = None

    for i, box in enumerate(boxes):
        x, y, w, h = box
        box_center_x = x + w / 2
        box_center_y = y + h / 2
        box_dist = dist((center_x, center_y), (box_center_x, box_center_y))

        if best_i is None or box_dist < best_dist:
            best_i = i
            best_dist = box_dist

    return best_i


def make_way_geometry(way: dict, nodes: dict[str, LatLon]) -> Sequence[LatLon]:
    return tuple(nodes[node_id] for node_id in way['nodes'])


def set_nice(value: int) -> None:
    os.nice(value - os.nice(0))


def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')


def format_eta(seconds: int) -> str:
    if seconds <= 0:
        return '0s'
    units = [('y', 60*60*24*365), ('d', 60*60*24), ('h', 60*60), ('m', 60), ('s', 1)]
    parts = []
    for unit, size in units:
        if seconds >= size:
            amount = seconds // size
            seconds -= amount * size
            parts.append(f'{amount}{unit}')
    return ' '.join(parts)
