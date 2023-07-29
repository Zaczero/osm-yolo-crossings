import pickle
import random
from math import ceil
from time import sleep
from typing import Generator, NamedTuple, Sequence

from numpy import arange

from box import Box
from config import CACHE_DIR, DB_GRID, SLEEP_AFTER_GRID_ITER
from grid_filter import is_grid_valid
from latlon import LatLon
from utils import meters_to_lat, meters_to_lon

_COUNTRY_BB = Box(LatLon(49.0, 14.0),
                  LatLon(55.0, 24.25) - LatLon(49.0, 14.0))

_DENSITY_Y = meters_to_lat(0.1)
_DENSITY_X = meters_to_lon(0.1, _COUNTRY_BB.center().lat)
_RATIO = _DENSITY_X / _DENSITY_Y

_GRID_SIZE_Y = 0.004 / 18
_GRID_SIZE_X = 0.004 / 18 * _RATIO
_MACRO_GRID_FACTOR = 16
_MACRO_GRID_SIZE_Y = _GRID_SIZE_Y * _MACRO_GRID_FACTOR
_MACRO_GRID_SIZE_X = _GRID_SIZE_X * _MACRO_GRID_FACTOR

_OVERLAP_PERCENT = 0.1

_GRID_OVERLAP_Y = _GRID_SIZE_Y * _OVERLAP_PERCENT
_GRID_OVERLAP_X = _GRID_SIZE_X * _OVERLAP_PERCENT
_MACRO_OVERLAP_Y = _MACRO_GRID_SIZE_Y * _OVERLAP_PERCENT
_MACRO_OVERLAP_X = _MACRO_GRID_SIZE_X * _OVERLAP_PERCENT


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


def _get_grid() -> Sequence[Cell]:
    cache_path = CACHE_DIR / 'grid.pkl'
    if cache_path.is_file():
        return pickle.loads(cache_path.read_bytes())

    p1 = _COUNTRY_BB.point
    p2 = _COUNTRY_BB.point + _COUNTRY_BB.size
    result = []

    for y in arange(p1.lat, p2.lat, _GRID_SIZE_Y * (1 - _OVERLAP_PERCENT)):
        for x in arange(p1.lon, p2.lon, _GRID_SIZE_X * (1 - _OVERLAP_PERCENT)):
            box = Box(LatLon(y, x), LatLon(_GRID_SIZE_Y, _GRID_SIZE_X))
            result.append(box)

    print(f'[GRID] # Generated {len(result)} cells')
    result = tuple(result)
    cache_path.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    return result


def iter_grid() -> Generator[Cell, None, None]:
    y_big_len = ceil(_COUNTRY_BB.size.lat / (_MACRO_GRID_SIZE_Y - _MACRO_OVERLAP_Y))
    x_big_len = ceil(_COUNTRY_BB.size.lon / (_MACRO_GRID_SIZE_X - _MACRO_OVERLAP_X))
    y_small_len = ceil(_MACRO_GRID_SIZE_Y / (_GRID_SIZE_Y - _GRID_OVERLAP_Y))
    x_small_len = ceil(_MACRO_GRID_SIZE_X / (_GRID_SIZE_X - _GRID_OVERLAP_X))
    grid_size = y_big_len * x_big_len * y_small_len * x_small_len

    index = 0
    last_index = _get_last_index()
    if last_index > -1:
        print(f'[GRID] ⏭️ Resuming from index {last_index + 1}')

    for y_big in range(y_big_len):
        if index < last_index and index + x_big_len * y_small_len * x_small_len <= last_index:
            index += x_big_len * y_small_len * x_small_len
            continue

        for x_big in range(x_big_len):
            if index < last_index and index + y_small_len * x_small_len <= last_index:
                index += y_small_len * x_small_len
                continue

            box_big_offset = LatLon(y_big * (_MACRO_GRID_SIZE_Y - _MACRO_OVERLAP_Y),
                                    x_big * (_MACRO_GRID_SIZE_X - _MACRO_OVERLAP_X))
            box_big = Box(_COUNTRY_BB.point + box_big_offset, LatLon(_MACRO_GRID_SIZE_Y, _MACRO_GRID_SIZE_X))

            if not is_grid_valid(box_big):
                index += y_small_len * x_small_len
                continue

            for y_small in range(y_small_len):
                for x_small in range(x_small_len):
                    if index > last_index:
                        box_small_offset = LatLon(y_small * (_GRID_SIZE_Y - _GRID_OVERLAP_Y),
                                                  x_small * (_GRID_SIZE_X - _GRID_OVERLAP_X))
                        box_small = Box(box_big.point + box_small_offset, LatLon(_GRID_SIZE_Y, _GRID_SIZE_X))

                        if is_grid_valid(box_small):
                            progress = index / grid_size
                            print(f'[GRID] ☑️ Yield index {index} ({progress:.4%})')
                            yield Cell(index, box_small)

                    index += 1


def set_last_cell(cell: Cell | None) -> None:
    _set_last_index(cell.index if cell else -1)


def random_grid() -> Sequence[Box]:
    result = list(_get_grid())
    random.shuffle(result)
    return tuple(result)
