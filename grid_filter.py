import pickle
from math import floor
from typing import Iterable, NamedTuple, Sequence

from cachetools import TTLCache, cached
from numpy import arange
from rtree.index import Index

from box import Box
from config import CACHE_DIR
from latlon import LatLon
from overpass import query_buildings_roads
from utils import print_run_time

_STATE_GRID_SIZE = 0.2
_MIN_ROAD_NODES = 1


class GridFilterState(NamedTuple):
    buildings: Sequence[LatLon]
    roads: Sequence[LatLon]


def _make_index(elements: Sequence[LatLon]) -> Index:
    index = Index()
    for i, e in enumerate(elements):
        index.insert(i, (e.lat, e.lon, e.lat, e.lon))
    return index


@cached(TTLCache(64, ttl=3600))
def _load_index(lat: float, lon: float) -> Index:
    cache_path = CACHE_DIR / f'FilterState_{lat:.2f}_{lon:.2f}.pkl'
    if cache_path.is_file():
        state = pickle.loads(cache_path.read_bytes())
    else:
        with print_run_time(f'Query grid filter: ({lat:.2f}, {lon:.2f})'):
            buildings, roads = query_buildings_roads(Box(
                point=LatLon(lat, lon),
                size=LatLon(_STATE_GRID_SIZE, _STATE_GRID_SIZE)))

        state = GridFilterState(buildings, roads)
        cache_path.write_bytes(pickle.dumps(state))
    return _make_index(state.roads)


def _iter_indexes(box: Box) -> Iterable[Index]:
    lat_start = floor(box.point.lat / _STATE_GRID_SIZE) * _STATE_GRID_SIZE
    lon_start = floor(box.point.lon / _STATE_GRID_SIZE) * _STATE_GRID_SIZE
    lat_end = box.point.lat + box.size.lat
    lon_end = box.point.lon + box.size.lon
    for lat in arange(lat_start, lat_end, _STATE_GRID_SIZE):
        for lon in arange(lon_start, lon_end, _STATE_GRID_SIZE):
            yield _load_index(lat, lon)


def is_grid_valid(box: Box) -> bool:
    sum_road_nodes = 0

    for road_index in _iter_indexes(box):
        roads = road_index.intersection((
            box.point.lat,
            box.point.lon,
            box.point.lat + box.size.lat,
            box.point.lon + box.size.lon))

        sum_road_nodes += sum(1 for _ in roads)
        if sum_road_nodes >= _MIN_ROAD_NODES:
            return True

    return False
