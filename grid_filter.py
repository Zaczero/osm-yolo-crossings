import pickle
from math import floor
from typing import NamedTuple, Sequence

from cachetools import TTLCache, cached
from numpy import arange
from rtree.index import Index

from box import Box
from config import CACHE_DIR
from latlon import LatLon
from overpass import query_buildings_roads
from utils import print_run_time

_STATE_GRID_SIZE = 0.2
# _MIN_BUILDINGS = 2
_MIN_ROADS = 1  # road nodes


def _make_index(elements: Sequence[LatLon]) -> Index:
    index = Index()
    for i, e in enumerate(elements):
        index.insert(i, (e.lat, e.lon, e.lat, e.lon))
    return index


class GridFilterState(NamedTuple):
    buildings: Sequence[LatLon]
    roads: Sequence[LatLon]

    @cached(TTLCache(64, ttl=3600))
    def get_index(self) -> tuple[Index, Index]:
        return _make_index(self.buildings), _make_index(self.roads)


@cached(TTLCache(64, ttl=3600))
def _load_state_single(lat: float, lon: float) -> GridFilterState:
    cache_path = CACHE_DIR / f'FilterState_{lat:.2f}_{lon:.2f}.pkl'
    if cache_path.is_file():
        return pickle.loads(cache_path.read_bytes())
    else:
        with print_run_time(f'Query grid filter: ({lat:.2f}, {lon:.2f})'):
            buildings, roads = query_buildings_roads(Box(
                point=LatLon(lat, lon),
                size=LatLon(_STATE_GRID_SIZE, _STATE_GRID_SIZE)))

        state = GridFilterState(buildings, roads)
        cache_path.write_bytes(pickle.dumps(state))
        return state


def _load_state(box: Box) -> Sequence[GridFilterState]:
    result = []

    for lat in arange(floor(box.point.lat / _STATE_GRID_SIZE) * _STATE_GRID_SIZE, box.point.lat + box.size.lat, _STATE_GRID_SIZE):
        for lon in arange(floor(box.point.lon / _STATE_GRID_SIZE) * _STATE_GRID_SIZE, box.point.lon + box.size.lon, _STATE_GRID_SIZE):
            result.append(_load_state_single(lat, lon))

    return tuple(result)


def is_grid_valid(box: Box) -> bool:
    # total_buildings = 0
    total_roads = 0

    for state in _load_state(box):
        building_index, road_index = state.get_index()

        # if total_buildings < _MIN_BUILDINGS:
        #     buildings = building_index.intersection(
        #         (box.point.lat, box.point.lon, box.point.lat + box.size.lat, box.point.lon + box.size.lon))
        #     total_buildings += sum(1 for _ in buildings)

        if total_roads < _MIN_ROADS:
            roads = road_index.intersection(
                (box.point.lat, box.point.lon, box.point.lat + box.size.lat, box.point.lon + box.size.lon))
            total_roads += sum(1 for _ in roads)

        if total_roads >= _MIN_ROADS:
            return True

    return False
