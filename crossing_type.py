from enum import Enum
from typing import Self


class CrossingType(Enum):
    UNKNOWN = 0
    UNCONTROLLED = 1
    TRAFFIC_SIGNALS = 2

    @staticmethod
    def make_tags(crossing_type: Self) -> dict[str, str]:
        if crossing_type == CrossingType.UNCONTROLLED:
            return {
                'highway': 'crossing',
                'crossing': 'uncontrolled',
                'crossing:markings': 'zebra',
            }
        elif crossing_type == CrossingType.TRAFFIC_SIGNALS:
            return {
                'highway': 'crossing',
                'crossing': 'traffic_signals',
                'crossing:markings': 'zebra',
            }
        else:
            return {
                'highway': 'crossing',
                'crossing:markings': 'zebra',
            }
