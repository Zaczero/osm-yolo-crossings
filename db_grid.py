import pickle
import random
from functools import reduce
from math import ceil
from operator import mul
from time import time
from typing import Generator, NamedTuple, Sequence

from numpy import arange

from box import Box
from config import CACHE_DIR, DB_GRID, GRID_OVERLAP
from grid_filter import is_grid_valid
from latlon import LatLon
from utils import format_eta, meters_to_lat, meters_to_lon

_COUNTRY_BB = Box(LatLon(49.0, 14.0),
                  LatLon(55.0, 24.25) - LatLon(49.0, 14.0))

_DENSITY_Y = meters_to_lat(0.1)
_DENSITY_X = meters_to_lon(0.1, _COUNTRY_BB.center().lat)
_RATIO = _DENSITY_X / _DENSITY_Y

_GRID_SIZE_Y = 0.004 / 18
_GRID_SIZE_X = 0.004 / 18 * _RATIO
_GRID_FACTORS = (4, 4, 4, 4, 4, 1)

_OVERLAP_Y = _GRID_SIZE_Y * GRID_OVERLAP
_OVERLAP_X = _GRID_SIZE_X * GRID_OVERLAP


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

    for y in arange(p1.lat, p2.lat, _GRID_SIZE_Y * (1 - GRID_OVERLAP)):
        for x in arange(p1.lon, p2.lon, _GRID_SIZE_X * (1 - GRID_OVERLAP)):
            box = Box(LatLon(y, x), LatLon(_GRID_SIZE_Y, _GRID_SIZE_X))
            result.append(box)

    print(f'[GRID] # Generated {len(result)} cells')
    result = tuple(result)
    cache_path.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    return result


class GridParams(NamedTuple):
    len_y: int
    len_x: int
    size: LatLon
    size_index: int = 1


def iter_grid() -> Generator[Cell, None, None]:
    grid_params: list[GridParams] = []
    total_factor = reduce(mul, _GRID_FACTORS)
    prev_size = _COUNTRY_BB.size

    for factor in _GRID_FACTORS:
        size_y = _GRID_SIZE_Y * total_factor
        size_x = _GRID_SIZE_X * total_factor
        size = LatLon(size_y, size_x)
        len_y = ceil(prev_size.lat / (size.lat - _OVERLAP_Y))
        len_x = ceil(prev_size.lon / (size.lon - _OVERLAP_X))
        grid_params.append(GridParams(len_y, len_x, size))
        total_factor /= factor
        prev_size = size

    for layer_index in range(len(grid_params) - 1):
        reduce_iter = (param_.len_y * param_.len_x for param_ in grid_params[layer_index + 1:])
        size_index = reduce(mul, reduce_iter, 1)
        grid_params[layer_index] = grid_params[layer_index]._replace(size_index=size_index)

    grid_params = tuple(grid_params)
    grid_size_index = grid_params[0].len_y * grid_params[0].len_x * grid_params[0].size_index

    next_index = 0
    last_index = _get_last_index()
    if last_index > -1:
        print(f'[GRID] ⏭️ Resume from last index {last_index}')

    start_time = time()

    def traverse(layer_index: int, parent_box: Box):
        nonlocal next_index

        param = grid_params[layer_index]

        for y in range(param.len_y):
            for x in range(param.len_x):
                offset = LatLon(y * (param.size.lat - _OVERLAP_Y), x * (param.size.lon - _OVERLAP_X))
                box = Box(parent_box.point + offset, param.size)

                if next_index < last_index and next_index + param.size_index <= last_index:
                    next_index += param.size_index
                    continue

                if not is_grid_valid(box):
                    next_index += param.size_index
                    continue

                if layer_index + 1 < len(grid_params):
                    yield from traverse(layer_index + 1, box)
                else:
                    if next_index > last_index:
                        progress = next_index / grid_size_index
                        elapsed_time = time() - start_time
                        eta = int((grid_size_index - next_index) / (next_index - last_index) * elapsed_time)
                        print(f'[{progress:.4%}] Yield index {next_index} - ETA: {format_eta(eta)}')
                        yield Cell(next_index, box)
                    next_index += 1

    yield from traverse(0, _COUNTRY_BB)


def set_last_cell_index(cell_index: int | None) -> None:
    last_index = cell_index or -1
    print(f'[GRID] ⏭️ Set last index to {last_index}')
    _set_last_index(last_index)


def random_grid() -> Sequence[Box]:
    result = list(_get_grid())
    random.shuffle(result)
    return tuple(result)
