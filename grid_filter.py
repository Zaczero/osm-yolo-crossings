import pickle
from math import floor
from typing import Iterable, NamedTuple, Sequence

from cachetools import TTLCache, cached
from numpy import arange
from rtree.index import Index

from box import Box
from config import CACHE_DIR, GRID_FILTER_BUILDING_DISTANCE
from latlon import LatLon
from overpass import query_buildings_roads
from utils import print_run_time

_STATE_GRID_SIZE = 0.2


class GridFilterState(NamedTuple):
    lat: float
    lon: float
    buildings: Sequence[LatLon]
    roads: Sequence[LatLon]

    def make_indexes(self) -> Sequence[Index]:
        def make_index(name: str, boxes: Sequence[Box]) -> Index:
            path = CACHE_DIR / f'FilterStateIndex_{name}_{self.lat:.2f}_{self.lon:.2f}'
            dat_path = path.with_name(path.name + '.dat')
            idx_path = path.with_name(path.name + '.idx')

            if not dat_path.is_file() or not idx_path.is_file():
                dat_path.unlink(missing_ok=True)
                idx_path.unlink(missing_ok=True)
                temp_path = idx_path.with_name(path.name + '.temp')
                temp_dat_path = temp_path.with_name(temp_path.name + '.dat')
                temp_dat_path.unlink(missing_ok=True)
                temp_idx_path = temp_path.with_name(temp_path.name + '.idx')
                temp_idx_path.unlink(missing_ok=True)
                index = Index(str(temp_path))

                for i, box in enumerate(boxes):
                    p1 = box.point
                    p2 = box.point + box.size
                    index.insert(i, (p1.lat, p1.lon, p2.lat, p2.lon))

                index.close()
                temp_dat_path.rename(dat_path)
                temp_idx_path.rename(idx_path)

            return Index(str(path))

        buildings_boxes = tuple(map(
            lambda p: Box(p, LatLon(0, 0)).extend(meters=GRID_FILTER_BUILDING_DISTANCE),
            self.buildings))

        roads_boxes = tuple(map(
            lambda p: Box(p, LatLon(0, 0)),
            self.roads))

        return make_index('buildings', buildings_boxes), make_index('roads', roads_boxes)


@cached(TTLCache(64, ttl=3600))
def _load_indexes(lat: float, lon: float) -> Sequence[Index]:
    cache_path = CACHE_DIR / f'FilterState_{lat:.2f}_{lon:.2f}.pkl'
    if cache_path.is_file():
        state = pickle.loads(cache_path.read_bytes())
    else:
        with print_run_time(f'Query grid filter: ({lat:.2f}, {lon:.2f})'):
            query_bbox = Box(
                point=LatLon(lat, lon),
                size=LatLon(_STATE_GRID_SIZE, _STATE_GRID_SIZE)) \
                .extend(meters=GRID_FILTER_BUILDING_DISTANCE)
            buildings, roads = query_buildings_roads(query_bbox)

        state = GridFilterState(lat, lon, buildings, roads)
        cache_path.write_bytes(pickle.dumps(state))
    return state.make_indexes()


def _iter_indexes(box: Box) -> Iterable[Sequence[Index]]:
    lat_start = floor(box.point.lat / _STATE_GRID_SIZE) * _STATE_GRID_SIZE
    lon_start = floor(box.point.lon / _STATE_GRID_SIZE) * _STATE_GRID_SIZE
    lat_end = box.point.lat + box.size.lat
    lon_end = box.point.lon + box.size.lon
    for lat in arange(lat_start, lat_end, _STATE_GRID_SIZE):
        for lon in arange(lon_start, lon_end, _STATE_GRID_SIZE):
            yield _load_indexes(lat, lon)


def is_grid_valid(box: Box) -> bool:
    p1 = box.point
    p2 = box.point + box.size

    for indexes in _iter_indexes(box):
        # true when all index layers have at least one intersection
        if all(any(i.intersection((p1.lat, p1.lon, p2.lat, p2.lon))) for i in indexes):
            return True

    return False
