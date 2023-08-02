import httpx
import numpy as np
from skimage import img_as_float
from skimage.io import imread
from tenacity import retry, stop_after_delay, wait_exponential

from box import Box
from config import RETRY_TIME_LIMIT
from utils import http_headers


@retry(wait=wait_exponential(), stop=stop_after_delay(RETRY_TIME_LIMIT))
def fetch_orto(box: Box, resolution: int) -> np.ndarray | None:
    assert resolution <= 4096, 'This resolution is not supported by the WMS service'

    r = httpx.get('https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution', params={
        'LAYERS': 'Raster',
        'STYLES': 'default',
        'FORMAT': 'image/jpeg',
        'CRS': 'EPSG:4326',
        'WIDTH': resolution,
        'HEIGHT': resolution,
        'BBOX': str(box),
        'VERSION': '1.3.0',
        'SERVICE': 'WMS',
        'REQUEST': 'GetMap',
    }, headers=http_headers(), timeout=120)

    r.raise_for_status()

    img = imread(r.content, plugin='imageio')

    if img.min() == img.max():
        return None

    return img_as_float(img)
