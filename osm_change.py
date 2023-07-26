from itertools import chain
from typing import Iterable

import xmltodict

from budynki import Building
from config import CHANGESET_ID_PLACEHOLDER, CREATED_BY


def _initialize_osm_change_structure() -> dict:
    return {
        'osmChange': {
            '@version': 0.6,
            '@generator': CREATED_BY,
            'create': {
                'node': [],
                'way': [],
            }
        }
    }


def create_buildings_change(buildings: Iterable[Building]) -> str:
    result = _initialize_osm_change_structure()
    create_nodes: list = result['osmChange']['create']['node']
    create_ways: list = result['osmChange']['create']['way']

    last_id = 0

    for building in buildings:
        point_idx_to_id = []

        for p in building.polygon.points[:-1]:
            last_id -= 1
            point_idx_to_id.append(last_id)
            create_nodes.append({
                '@id': last_id,
                '@changeset': CHANGESET_ID_PLACEHOLDER,
                '@version': 1,
                '@lat': p.lat,
                '@lon': p.lon,
            })

        last_id -= 1
        create_ways.append({
            '@id': last_id,
            '@changeset': CHANGESET_ID_PLACEHOLDER,
            '@version': 1,
            'nd': [
                {'@ref': point_idx_to_id[i]}
                for i in chain(range(len(point_idx_to_id)), (0,))
            ],
            'tag': [
                {'@k': k, '@v': v}
                for k, v in building.tags.items()
            ]
        })

    return xmltodict.unparse(result)
