from typing import Iterable, Sequence

from box import Box
from latlon import LatLon


def transform_rad_to_px(points: Iterable[LatLon], img_box: Box, img_shape: tuple[int, int]) -> Sequence[tuple[int, int]]:
    y_res = img_box.size.lat / img_shape[0]
    x_res = img_box.size.lon / img_shape[1]

    result = []

    for point in points:
        y = round((img_box.point.lat + img_box.size.lat - point.lat) / y_res)
        x = round((point.lon - img_box.point.lon) / x_res)
        result.append((y, x))

    return tuple(result)


def transform_px_to_rad(points: Iterable[tuple[int, int]], img_box: Box, img_shape: tuple[int, int]) -> Sequence[LatLon]:
    y_res = img_box.size.lat / img_shape[0]
    x_res = img_box.size.lon / img_shape[1]

    result = []

    for point in points:
        lat = img_box.point.lat + img_box.size.lat - point[0] * y_res
        lon = img_box.point.lon + point[1] * x_res
        result.append(LatLon(lat, lon))

    return tuple(result)
