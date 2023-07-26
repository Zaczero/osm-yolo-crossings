from enum import Enum

import httpx
import numpy as np
from skimage import img_as_float
from skimage.io import imread
from tenacity import retry, stop_after_attempt, wait_exponential

from box import Box
from config import YOLO_MODEL_RESOLUTION
from utils import http_headers

_RESOLUTION = 4032  # 224 * 18
assert _RESOLUTION <= 4096, 'This resolution is not supported by the WMS service'
assert _RESOLUTION % YOLO_MODEL_RESOLUTION == 0, 'The resolution must be divisible by the model resolution'


class FetchMode(Enum):
    FAST = 1
    QUALITY = 2


@retry(wait=wait_exponential(), stop=stop_after_attempt(5))
def fetch_orto(box: Box, mode: FetchMode) -> np.ndarray | None:
    r = httpx.get('https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution', params={
        'LAYERS': 'Raster',
        'STYLES': 'default',
        'FORMAT': 'image/png' if mode == FetchMode.QUALITY else 'image/jpeg',
        'CRS': 'EPSG:4326',
        'WIDTH': _RESOLUTION,
        'HEIGHT': _RESOLUTION,
        'BBOX': str(box),
        'VERSION': '1.3.0',
        'SERVICE': 'WMS',
        'REQUEST': 'GetMap',
    }, headers=http_headers(), timeout=200)

    r.raise_for_status()

    img = imread(r.content, plugin='imageio')

    if img.min() == img.max():
        return None

    return img_as_float(img)
