from base64 import urlsafe_b64encode
from hashlib import sha256
from typing import NamedTuple, Sequence

from box import Box
from latlon import LatLon


class Polygon(NamedTuple):
    points: Sequence[LatLon]

    # def extend(self, meters: float) -> Self:
    #     points_arr = np.array(self.points)
    #     edge_vectors = np.roll(points_arr, -1, axis=0) - points_arr
    #     edge_vectors[-1] = points_arr[0] - points_arr[-1]
    #     normal_vectors = np.stack([-edge_vectors[:, 1], edge_vectors[:, 0]], axis=-1)
    #     normal_vectors /= np.linalg.norm(normal_vectors, axis=1, keepdims=True)
    #     return Polygon(tuple(points_arr + normal_vectors * meters_to_lat(meters)))

    def get_bounding_box(self) -> Box:
        min_lat = min(p.lat for p in self.points)
        max_lat = max(p.lat for p in self.points)
        min_lon = min(p.lon for p in self.points)
        max_lon = max(p.lon for p in self.points)

        return Box(
            point=LatLon(min_lat, min_lon),
            size=LatLon(max_lat - min_lat, max_lon - min_lon),
        )

    def unique_id(self) -> str:
        precision = 0.00003
        rounded_points = (
            f'{(round(p.lat / precision) * precision):.5f},{(round(p.lon / precision) * precision):.5f}'
            for p in self.points)
        return ';'.join(sorted(rounded_points))

    def unique_id_hash(self, size: int = 6) -> str:
        result = sha256(self.unique_id().encode(), usedforsecurity=False).digest()
        result = result[:size]
        return urlsafe_b64encode(result).decode()
