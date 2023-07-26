from math import atan2

from shapely.geometry import LineString


class Line:
    def __init__(self, line, *, expand_bbox: float = 0):
        self.line = LineString(line)
        self.angle = self._get_angle()
        self.bbox = self._get_expanded_bbox(expand_bbox)

    def _get_angle(self) -> float:
        (y1, x1), (y2, x2) = self.line.coords
        if x1 > x2:
            y1, x1, y2, x2 = y2, x2, y1, x1
        return atan2(y2 - y1, x2 - x1)

    def _get_expanded_bbox(self, amount: float) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.line.bounds
        return (x1 - amount, y1 - amount, x2 + amount, y2 + amount)
