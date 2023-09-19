from time import time
from typing import Iterable, Sequence

from box import Box
from config import ADDED_SEARCH_RADIUS, MONGO_ADDED, SCORER_VERSION, VERSION
from latlon import LatLon


def contains_added(box: Box) -> bool:
    p1 = box.point
    p2 = box.point + box.size

    return MONGO_ADDED.find_one({
        "$and": [
            {"position": {
                "$geoIntersects": {
                    "$geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [p1.lon, p1.lat],
                                [p1.lon, p2.lat],
                                [p2.lon, p2.lat],
                                [p2.lon, p1.lat],
                                [p1.lon, p1.lat]
                            ]
                        ]
                    }
                }
            }},
            {"scorer_version": {"$gte": SCORER_VERSION}}
        ]
    }) is not None


def mask_not_added(positions: Iterable[LatLon]) -> Sequence[bool]:
    result = []

    for p in positions:
        doc = MONGO_ADDED.find_one({
            "$and": [
                {"position": {
                    "$nearSphere": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": (p.lon, p.lat)
                        },
                        "$maxDistance": ADDED_SEARCH_RADIUS
                    }
                }},
                {"$or": [
                    {"scorer_version": {"$gte": SCORER_VERSION}},
                    {"reason": "added"}
                ]}
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
