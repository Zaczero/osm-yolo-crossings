from math import cos, radians, sin, tan
from typing import NamedTuple, Self, Sequence

import numpy as np


class Polygon2(NamedTuple):
    points: Sequence[tuple[float, float]]

    @classmethod
    def from_str(cls, s: str) -> Self:
        points = []
        for point_str in s.split(';'):
            x, y = point_str.split(',')
            points.append((float(x), float(y)))
        return cls(tuple(points))

    def transform_and_bb(self, params: dict, shape: Sequence[int]) -> tuple[float, float, float, float]:
        center_x = shape[1] / 2
        center_y = shape[0] / 2

        points = tuple(list(p) for p in self.points)

        for p in points:
            # rotate
            theta = -radians(params['theta'])
            dx = p[0] - center_x
            dy = p[1] - center_y
            p[0] = dx * cos(theta) - dy * sin(theta) + center_x
            p[1] = dx * sin(theta) + dy * cos(theta) + center_y

            # translate
            p[0] -= params['tx']
            p[1] -= params['ty']

            # shear
            shear = -radians(params['shear'])
            dx = p[0] - center_x
            dy = p[1] - center_y
            p[0] = dx - sin(shear) * dy + center_x
            p[1] = cos(shear) * dy + center_y

            # scale
            dx = p[0] - center_x
            dy = p[1] - center_y
            dx /= params['zx']
            dy /= params['zy']
            p[0] = center_x + dx
            p[1] = center_y + dy

            # flip
            if params['flip_horizontal']:
                p[0] = shape[1] - p[0]
            if params['flip_vertical']:
                p[1] = shape[0] - p[1]

        # bounding box
        x_min = min(p[0] for p in points)
        x_max = max(p[0] for p in points)
        y_min = min(p[1] for p in points)
        y_max = max(p[1] for p in points)

        # clip
        x_min = np.clip(x_min, 0, shape[1])
        x_max = np.clip(x_max, 0, shape[1])
        y_min = np.clip(y_min, 0, shape[0])
        y_max = np.clip(y_max, 0, shape[0])

        x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

        return x, y, w, h
