from typing import NamedTuple, Self


class LatLon(NamedTuple):
    lat: float
    lon: float

    def __str__(self) -> str:
        return f'{self.lat:.8f},{self.lon:.8f}'

    def __add__(self, other) -> Self:
        return LatLon(self.lat + other.lat, self.lon + other.lon)

    def __sub__(self, other) -> Self:
        return LatLon(self.lat - other.lat, self.lon - other.lon)
