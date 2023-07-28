import random
from math import ceil
from time import sleep
from typing import Generator, NamedTuple, Sequence

from numpy import arange

from box import Box
from config import DB_GRID, SLEEP_AFTER_GRID_ITER
from grid_filter import is_grid_valid
from latlon import LatLon
from utils import meters_to_lat, meters_to_lon

_COUNTRY_BB = Box(LatLon(49.0, 14.0),
                  LatLon(55.0, 24.25) - LatLon(49.0, 14.0))

_DENSITY_Y = meters_to_lat(0.1)
_DENSITY_X = meters_to_lon(0.1, _COUNTRY_BB.center().lat)
_RATIO = _DENSITY_X / _DENSITY_Y

_GRID_SIZE_Y = 0.004
_GRID_SIZE_X = 0.004 * _RATIO

_OVERLAP_PERCENT = 0.1


class Cell(NamedTuple):
    index: int
    box: Box


def _get_last_index() -> int:
    doc = DB_GRID.get(doc_id=1)

    if doc is None:
        return -1

    return doc['index']


def _set_last_index(index: int) -> None:
    DB_GRID.upsert({'index': index}, lambda _: True)


def iter_grid() -> Generator[Cell, None, None]:
    last_index = _get_last_index()

    if last_index > -1:
        print(f'[GRID] ⏭️ Resuming from index {last_index + 1}')

    index = 0

    for y in range(ceil(_COUNTRY_BB.size.lat / _GRID_SIZE_Y)):
        for x in range(ceil(_COUNTRY_BB.size.lon / _GRID_SIZE_X)):
            if index > last_index:
                box = Box(
                    point=_COUNTRY_BB.point + LatLon(y * _GRID_SIZE_Y, x * _GRID_SIZE_X),
                    size=LatLon(_GRID_SIZE_Y, _GRID_SIZE_X))

                if is_grid_valid(box):
                    print(f'[GRID] ☑️ Yield index {index}')
                    yield Cell(index, box)

            index += 1


def set_last_cell(cell: Cell | None) -> None:
    _set_last_index(cell.index if cell else -1)


def random_grid() -> Sequence[Box]:
    country_bb_end = _COUNTRY_BB.point + _COUNTRY_BB.size
    result = []

    for y in arange(_COUNTRY_BB.point.lat, country_bb_end.lat, _GRID_SIZE_Y * (1 - _OVERLAP_PERCENT)):
        for x in arange(_COUNTRY_BB.point.lon, country_bb_end.lon, _GRID_SIZE_X * (1 - _OVERLAP_PERCENT)):
            box = Box(LatLon(y, x), LatLon(_GRID_SIZE_Y, _GRID_SIZE_X))
            result.append(box)

    print(f'[GRID] # Generated {len(result)} cells')
    random.shuffle(result)
    return tuple(result)
