from math import atan, cos, degrees, log, pi, radians, sinh, tan

import httpx
import numpy as np
from cachetools import TTLCache, cached
from skimage import img_as_float, transform
from skimage.io import imread
from tenacity import retry, stop_after_delay, wait_exponential

from box import Box
from config import RETRY_TIME_LIMIT
from latlon import LatLon
from utils import http_headers

_ZOOM = 19
_http = httpx.Client()


def _position_to_tile(p: LatLon) -> tuple[int, int]:
    n = 2 ** _ZOOM
    x = int((p.lon + 180.0) / 360.0 * n)
    y = int((1.0 - log(tan(radians(p.lat)) + (1 / cos(radians(p.lat)))) / pi) / 2.0 * n)
    return x, y


def _tile_to_position(x: int, y: int) -> LatLon:
    n = 2 ** _ZOOM
    lon = x / n * 360.0 - 180.0
    lat = atan(sinh(pi * (1 - 2 * y / n)))
    return LatLon(degrees(lat), lon)


def _crop_stitched_image(stitched_img: np.ndarray, stitched_p1: LatLon, stitched_p2: LatLon, box: Box):
    assert stitched_p1.lat <= box.point.lat <= stitched_p2.lat
    assert stitched_p1.lat <= box.point.lat + box.size.lat <= stitched_p2.lat
    assert stitched_p1.lon <= box.point.lon <= stitched_p2.lon
    assert stitched_p1.lon <= box.point.lon + box.size.lon <= stitched_p2.lon

    y_factor = stitched_img.shape[0] / (stitched_p2.lat - stitched_p1.lat)
    x_factor = stitched_img.shape[1] / (stitched_p2.lon - stitched_p1.lon)

    y1 = int((stitched_p2.lat - (box.point.lat + box.size.lat)) * y_factor)
    y2 = int((stitched_p2.lat - box.point.lat) * y_factor)

    x1 = int((box.point.lon - stitched_p1.lon) * x_factor)
    x2 = int((box.point.lon + box.size.lon - stitched_p1.lon) * x_factor)

    assert 0 <= y1 <= y2 <= stitched_img.shape[0]
    assert 0 <= x1 <= x2 <= stitched_img.shape[1]
    return stitched_img[y1:y2, x1:x2, :]


@cached(TTLCache(maxsize=32, ttl=3600))
@retry(wait=wait_exponential(max=1800), stop=stop_after_delay(RETRY_TIME_LIMIT))
def _fetch_wmts(x: int, y: int) -> np.ndarray | None:
    r = _http.get('https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMTS/StandardResolution', params={
        'SERVICE': 'WMTS',
        'REQUEST': 'GetTile',
        'VERSION': '1.0.0',
        'LAYER': 'ORTOFOTOMAPA',
        'STYLE': 'default',
        'FORMAT': 'image/jpeg',
        'tileMatrixSet': 'EPSG:3857',
        'tileMatrix': f'EPSG:3857:{_ZOOM}',
        'tileRow': y,
        'tileCol': x,
    }, headers=http_headers(), timeout=120)

    r.raise_for_status()

    if not r.content:
        return None

    img = imread(r.content, plugin='imageio')

    if img.min() == img.max():
        return None

    return img_as_float(img)


def fetch_orto(box: Box, resolution: int) -> np.ndarray | None:
    x_min, y_max = _position_to_tile(box.point)
    x_max, y_min = _position_to_tile(box.point + box.size)

    tiles = []

    for y in range(y_min, y_max + 1):
        row = []
        for x in range(x_min, x_max + 1):
            tile = _fetch_wmts(x, y)
            if tile is None:
                return None
            row.append(tile)
        tiles.append(row)

    stitched_img = np.vstack([np.hstack(row) for row in tiles])
    stitched_p1 = _tile_to_position(x_min, y_max + 1)
    stitched_p2 = _tile_to_position(x_max + 1, y_min)

    cropped_img = _crop_stitched_image(stitched_img, stitched_p1, stitched_p2, box)

    assert cropped_img.shape[0] <= resolution
    assert cropped_img.shape[1] <= resolution

    img = transform.resize(cropped_img, (resolution, resolution), order=3)

    return img_as_float(img)
