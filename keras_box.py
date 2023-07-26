from typing import NamedTuple, Sequence

import numpy as np

_MAX_BOXES = 20


class KerasBoundingBox(NamedTuple):
    x: float
    y: float
    w: float
    h: float


class KerasBoundingBoxes(NamedTuple):
    boxes: Sequence[KerasBoundingBox]
    classes: Sequence[int]

    def array(self) -> tuple[np.ndarray, np.ndarray]:
        padding = _MAX_BOXES - len(self.boxes)
        assert padding >= 0, f'Padding is negative: {padding}'

        if padding < _MAX_BOXES:
            boxes = np.array(self.boxes, float)
            classes = np.array(self.classes, int)

            boxes = np.pad(boxes, ((0, padding), (0, 0)), constant_values=-1)
            classes = np.pad(classes, (0, padding), constant_values=0)

            return boxes, classes

        else:
            boxes = np.full((_MAX_BOXES, 4), -1, dtype=float)
            classes = np.full((_MAX_BOXES,), 0, dtype=int)

            return boxes, classes
