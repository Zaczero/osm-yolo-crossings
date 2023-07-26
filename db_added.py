from time import time
from typing import Iterable, Sequence

from budynki import Building
from config import DB_ADDED, DB_ADDED_INDEX, SCORER_VERSION, VERSION


def _get_index() -> dict[str, int]:
    doc = DB_ADDED_INDEX.get(doc_id=1)

    if doc is None:
        return {}

    return doc['index']


def _set_index(index: dict[str, int]) -> None:
    DB_ADDED_INDEX.upsert({'index': index}, lambda _: True)


def filter_added(buildings: Iterable[Building]) -> Sequence[Building]:
    index = _get_index()

    def _is_added(building: Building) -> bool:
        unique_id = building.polygon.unique_id_hash(8)
        doc_id = index.get(unique_id, None)

        if doc_id is None:
            return False

        doc = DB_ADDED.get(doc_id=doc_id)
        return doc['scorer_version'] >= SCORER_VERSION

    return tuple(filter(lambda b: not _is_added(b), buildings))


def mark_added(buildings: Sequence[Building], **kwargs) -> Sequence[int]:
    unique_ids = tuple(b.polygon.unique_id_hash(8) for b in buildings)

    ids = DB_ADDED.insert_multiple({
        'timestamp': time(),
        'unique_id': unique_id,
        'location': tuple(building.polygon.points[0]),
        'app_version': VERSION,
        'scorer_version': SCORER_VERSION,
    } | kwargs for building, unique_id in zip(buildings, unique_ids))

    index = _get_index()

    for doc_id, unique_id in zip(ids, unique_ids):
        index[unique_id] = doc_id

    _set_index(index)
