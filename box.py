from typing import NamedTuple, Self

from latlon import LatLon
from utils import meters_to_lat, meters_to_lon


class Box(NamedTuple):
    point: LatLon
    size: LatLon

    def __str__(self) -> str:
        return f'{self.point.lat:.8f},{self.point.lon:.8f},{self.point.lat + self.size.lat:.8f},{self.point.lon + self.size.lon:.8f}'

    def __contains__(self, point: LatLon) -> bool:
        return (
            self.point.lat <= point.lat <= self.point.lat + self.size.lat and
            self.point.lon <= point.lon <= self.point.lon + self.size.lon
        )

    def extend(self, meters: float) -> Self:
        extend_lat = meters_to_lat(meters)
        extend_lon = meters_to_lon(meters, self.point.lat)

        return Box(
            point=LatLon(self.point.lat - extend_lat, self.point.lon - extend_lon),
            size=LatLon(self.size.lat + extend_lat * 2, self.size.lon + extend_lon * 2))

    def squarify(self) -> Self:
        # calculate the center of the bounding box
        center_lat = self.point.lat + self.size.lat / 2
        center_lon = self.point.lon + self.size.lon / 2

        density_y = meters_to_lat(0.1)
        density_x = meters_to_lon(0.1, center_lat)

        d_lat = self.size.lat
        d_lon = self.size.lon

        height = d_lat / density_y
        width = d_lon / density_x
        ratio = width / height

        if ratio > 1:
            d_lat = d_lat * ratio
        else:
            d_lon = d_lon / ratio

        return Box(
            point=LatLon(center_lat - d_lat / 2, center_lon - d_lon / 2),
            size=LatLon(d_lat, d_lon))

    def center(self) -> LatLon:
        return LatLon(self.point.lat + self.size.lat / 2, self.point.lon + self.size.lon / 2)
