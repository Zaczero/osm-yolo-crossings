from itertools import chain
from math import isclose
from typing import Sequence

import xmltodict
from shapely.geometry import LineString, Point

from config import CHANGESET_ID_PLACEHOLDER, CREATED_BY
from crossing_merger import CrossingMergeInstructions
from crossing_type import CrossingType
from latlon import LatLon
from openstreetmap import OpenStreetMap
from utils import print_run_time


def _merge_tags(e: dict, tags: dict, *, from_xml: bool) -> dict:
    if from_xml:
        e_tags = {t['@k']: t['@v'] for t in e.get('tag', [])}
    else:
        raise NotImplementedError

    return e_tags | tags


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
    with print_run_time('Fetching latest nodes data'):
        fetch_nodes = osm.get_nodes(
            tuple(chain.from_iterable(i.to_nodes_ids for i in instructions)))
        fetch_nodes = {n['@id']: n for n in fetch_nodes}

    with print_run_time('Fetching latest ways data'):
        fetch_ways = {}
        fetch_ways_geom = {}
        for inst in instructions:
            for way_inst in inst.to_ways_inst:
                fetch_way_full = osm.get_way_full(way_inst.way_id)
                fetch_way = fetch_way_full['way'][0]
                fetch_ways[fetch_way['@id']] = fetch_way

                for node in fetch_way_full['node']:
                    fetch_ways_geom[node['@id']] = node

    # increment metadata
    for node in fetch_nodes.values():
        node['@changeset'] = CHANGESET_ID_PLACEHOLDER

    for way in fetch_ways.values():
        way['@changeset'] = CHANGESET_ID_PLACEHOLDER

    result = _initialize_osm_change_structure()
    create_node: list = result['osmChange']['create']['node']
    last_id = 0

    for inst in instructions:
        tags = CrossingType.make_tags(inst.crossing_type)

        for node_id in inst.to_nodes_ids:
            node = fetch_nodes[node_id]
            node['tag'] = tuple(
                {'@k': k, '@v': v}
                for k, v in _merge_tags(node, tags, from_xml=True).items()
            )

        for way_inst in inst.to_ways_inst:
            way = fetch_ways[way_inst.way_id]

            last_id -= 1
            new_node = {
                '@id': last_id,
                '@changeset': CHANGESET_ID_PLACEHOLDER,
                '@version': 1,
                '@lat': way_inst.position.lat,
                '@lon': way_inst.position.lon,
                'tag': tuple(
                    {'@k': k, '@v': v}
                    for k, v in tags.items()
                ),
            }

            for i, (node1_ref, node2_ref) in enumerate(zip(way['nd'], way['nd'][1:])):
                node1_id = node1_ref['@ref']
                node2_id = node2_ref['@ref']
                node1 = fetch_nodes.get(node1_id, fetch_ways_geom.get(node1_id))
                node2 = fetch_nodes.get(node2_id, fetch_ways_geom.get(node2_id))
                p1 = LatLon(node1['@lat'], node1['@lon'])
                p2 = LatLon(node2['@lat'], node2['@lon'])

                if isclose(LineString([p1, p2]).distance(Point(way_inst.position)), 0, abs_tol=1e-8):
                    create_node.append(new_node)
                    fetch_ways_geom[new_node['@id']] = new_node
                    way['nd'].insert(i + 1, {'@ref': new_node['@id']})
                    break

    result['osmChange']['modify']['node'] = tuple(fetch_nodes.values())
    result['osmChange']['modify']['way'] = tuple(fetch_ways.values())

    return xmltodict.unparse(result)
