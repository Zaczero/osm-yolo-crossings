from time import time
from typing import Sequence

from sklearn.neighbors import BallTree
from tinydb import Query

from config import ADDED_POSITION_SEARCH, DB_ADDED, SCORER_VERSION, VERSION
from latlon import LatLon


def filter_added(positions: Sequence[LatLon]) -> Sequence[LatLon]:
    docs = DB_ADDED.get(Query().scorer_version >= SCORER_VERSION)
    db_positions = tuple(LatLon(*doc['position']) for doc in docs)
    tree = BallTree(db_positions, metric='haversine')
    query = tree.query_radius(tuple(positions), ADDED_POSITION_SEARCH, count_only=True)
    return tuple(p for q, p in zip(query, positions) if q == 0)


def mark_added(positions: Sequence[LatLon], **kwargs) -> Sequence[int]:
    return DB_ADDED.insert_multiple({
        'timestamp': time(),
        'position': tuple(p),
        'app_version': VERSION,
        'scorer_version': SCORER_VERSION,
    } | kwargs for p in positions)
