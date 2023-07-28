from time import time
from typing import Sequence

from sklearn.neighbors import BallTree
from tinydb import Query

from config import ADDED_POSITION_SEARCH, DB_ADDED, SCORER_VERSION, VERSION
from latlon import LatLon
from utils import meters_to_lat

_tree: BallTree | None = None


def _get_tree() -> BallTree:
    global _tree
    if _tree is None:
        docs = DB_ADDED.search(Query().scorer_version >= SCORER_VERSION)

        if not docs:
            positions = ((0, 0),)
        else:
            positions = tuple(LatLon(*doc['position']) for doc in docs)

        _tree = BallTree(positions, metric='haversine')
    return _tree


def filter_not_added(positions: Sequence[LatLon]) -> Sequence[bool]:
    query = _get_tree().query_radius(positions, meters_to_lat(ADDED_POSITION_SEARCH), count_only=True)
    return tuple(q == 0 for q in query)


def mark_added(positions: Sequence[LatLon], **kwargs) -> Sequence[int]:
    if not positions:
        return tuple()

    # clear cache
    global _tree
    _tree = None

    return DB_ADDED.insert_multiple({
        'timestamp': time(),
        'position': tuple(p),
        'app_version': VERSION,
        'scorer_version': SCORER_VERSION,
    } | kwargs for p in positions)
