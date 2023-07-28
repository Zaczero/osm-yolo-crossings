from itertools import chain
from typing import Sequence

import xmltodict

from config import CHANGESET_ID_PLACEHOLDER, CREATED_BY
from crossing_merger import CrossingMergeInstructions
from crossing_type import CrossingType
from openstreetmap import OpenStreetMap


def _merge_tags(e: dict, tags: dict) -> dict:
    return e.get('tags', {}) | tags


def _initialize_osm_change_structure() -> dict:
    return {
        'osmChange': {
            '@version': 0.6,
            '@generator': CREATED_BY,
            'create': {
                'node': [],
            },
            'modify': {
                'node': [],
                'way': [],
            },
        }
    }


def create_instructed_change(instructions: Sequence[CrossingMergeInstructions]) -> str:
    osm = OpenStreetMap()

    # fetch latest data
    fetch_nodes = osm.get_nodes(tuple(chain.from_iterable(
        i.to_nodes_ids for i in instructions)))
    fetch_nodes = {n['@id']: n for n in fetch_nodes}

    fetch_ways = osm.get_ways(tuple(chain.from_iterable(
        (i.way_id for i in inst.to_ways_inst) for inst in instructions)))
    fetch_ways = {w['@id']: w for w in fetch_ways}

    # increment metadata
    for node in fetch_nodes.values():
        node['@changeset'] = CHANGESET_ID_PLACEHOLDER
        node['@version'] += 1

    for way in fetch_ways.values():
        way['@changeset'] = CHANGESET_ID_PLACEHOLDER
        way['@version'] += 1

    result = _initialize_osm_change_structure()
    create_node: list = result['osmChange']['create']['node']
    last_id = 0

    for inst in instructions:
        tags = CrossingType.make_tags(inst.crossing_type)

        for node_id in inst.to_nodes_ids:
            node = fetch_nodes[node_id]
            node['tag'] = tuple(
                {'@k': k, '@v': v}
                for k, v in _merge_tags(node, tags).items()
            )

        for way_inst in inst.to_ways_inst:
            way = fetch_ways[way_inst.way_id]

            try:
                after_index = way['nd'].index({'@ref': way_inst.after_node_id})
                before_index = way['nd'].index({'@ref': way_inst.before_node_id})
            except ValueError:
                continue

            if after_index > before_index:
                after_index, before_index = before_index, after_index

            if after_index + 1 != before_index:
                continue

            last_id -= 1
            create_node.append({
                '@id': last_id,
                '@changeset': CHANGESET_ID_PLACEHOLDER,
                '@version': 1,
                '@lat': way_inst.position.lat,
                '@lon': way_inst.position.lon,
                'tag': tuple(
                    {'@k': k, '@v': v}
                    for k, v in tags.items()
                ),
            })

            way['nd'].insert(
                after_index + 1,
                {'@ref': last_id},
            )

    result['osmChange']['modify']['node'] = tuple(fetch_nodes.values())
    result['osmChange']['modify']['way'] = tuple(fetch_ways.values())

    return xmltodict.unparse(result)
