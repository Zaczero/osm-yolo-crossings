from time import time
from typing import Iterable, Sequence

from config import ADDED_SEARCH_RADIUS, MONGO_ADDED, SCORER_VERSION, VERSION
from latlon import LatLon


def mask_not_added(positions: Iterable[LatLon]) -> Sequence[bool]:
    result = []

    for p in positions:
        doc = MONGO_ADDED.find_one({
            "$and": [
                {"$or": [
                    {"scorer_version": {"$gte": SCORER_VERSION}},
                    {"reason": "added"}
                ]},
                {"position": {
                    "$nearSphere": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": (p.lon, p.lat)
                        },
                        "$maxDistance": ADDED_SEARCH_RADIUS
                    }
                }}
            ]
        })

        result.append(doc is None)

    return tuple(result)


def mark_added(positions: Sequence[LatLon], **kwargs) -> Sequence[int]:
    if not positions:
        return ()

    return MONGO_ADDED.insert_many([{
        'timestamp': time(),
        'position': (p.lon, p.lat),
        'app_version': VERSION,
        'scorer_version': SCORER_VERSION,
        **kwargs,
    } for p in positions]).inserted_ids
