from datetime import UTC, datetime, timedelta
from itertools import pairwise
from time import time
from typing import Iterable, NamedTuple, Sequence

import httpx
from tenacity import retry, stop_after_delay, wait_exponential

from box import Box
from config import (GRID_FILTER_ROAD_INTERPOLATE, OVERPASS_API_INTERPRETER,
                    RETRY_TIME_LIMIT, SEARCH_RELATION)
from latlon import LatLon
from utils import haversine_distance, http_headers


class QueriedCrossing(NamedTuple):
    position: LatLon
    tags: dict[str, str]
    bicycle: bool


class QueriedRoadsAndCrossings(NamedTuple):
    roads: Sequence[dict]
    crossings: Sequence[dict]
    paths: Sequence[dict]
    nodes: dict[int, LatLon]


def _build_elements_query(timeout: int, query: str) -> str:
    return (
        f'[out:json][timeout:{timeout}];'
        f'rel({SEARCH_RELATION});'
        f'map_to_area->.a;'
        f'{query}(area.a);'
        f'out ids center qt;'
    )


def _build_specific_crossings_query(box: Box, timeout: int, specific: str) -> str:
    return (
        f'[out:json][timeout:{timeout}][bbox:{box}];'
        f'nw[highway=crossing][crossing{specific}];'
        f'out body center qt;'
    )


def _build_buildings_roads_query(box: Box, timeout: int) -> str:
    return (
        f'[out:json][timeout:{timeout}][bbox:{box}];'
        f'rel({SEARCH_RELATION});'
        f'map_to_area->.a;'
        f'way[building](area.a);'
        f'out ids center qt;'
        f'out count;'
        f'way[highway](area.a);'
        f'out body qt;'
        f'out count;'
        f'>;'
        f'out skel qt;'
        f'out count;'
    )


def _build_roads_query(boxes: Sequence[Box], timeout: int) -> str:
    return (
        f'[out:json][timeout:{timeout}];' +
        f''.join(
            f'way[highway]({box});'
            f'out body qt;'
            f'>;'
            f'out body qt;'
            f'out count;'
            for box in boxes
        )
    )


def _split_by_count(elements: Iterable[dict]) -> list[list[dict]]:
    result = []
    current_split = []

    for e in elements:
        if e['type'] == 'count':
            result.append(current_split)
            current_split = []
        else:
            current_split.append(e)

    assert not current_split, 'Last element must be count type'
    return result


def _extract_center(elements: Sequence[dict]) -> None:
    for e in elements:
        if 'center' in e:
            e |= e['center']
            del e['center']


def _is_bicycle(element: dict) -> bool:
    tags = element.get('tags', {})

    return (
        tags.get('bicycle', 'no') != 'no' or
        tags.get('crossing:markings', '') == 'dots'
    )


def _is_road(element: dict) -> bool:
    tags = element.get('tags', {})

    return (
        tags.get('highway', '') in {
            'residential',
            'service',  # https://www.openstreetmap.org/way/444251815
            'unclassified',
            'tertiary',
            'secondary',
            'primary',
            'living_street',
            'road',
        } and
        tags.get('area', 'no') == 'no'
    )


def _is_path(element: dict) -> bool:
    tags = element.get('tags', {})

    return (
        tags.get('highway', '') in {
            'path',
            'footway',
            'cycleway',
            'pedestrian',
            'steps',
        } and
        tags.get('area', 'no') == 'no'
    )


def _is_crossing(element: dict) -> bool:
    tags = element.get('tags', {})

    return tags.get('highway', '') == 'crossing'


@retry(wait=wait_exponential(), stop=stop_after_delay(RETRY_TIME_LIMIT))
def query_elements_position(query: str) -> Sequence[LatLon]:
    timeout = 180
    query = _build_elements_query(timeout, query)

    r = httpx.post(OVERPASS_API_INTERPRETER, data={'data': query}, headers=http_headers(), timeout=timeout * 2)
    r.raise_for_status()

    elements = r.json()['elements']
    _extract_center(elements)

    result = tuple(LatLon(e['lat'], e['lon']) for e in elements)

    return result


@retry(wait=wait_exponential(), stop=stop_after_delay(RETRY_TIME_LIMIT))
def query_specific_crossings(box: Box, specific: str) -> Sequence[QueriedCrossing]:
    timeout = 180
    query = _build_specific_crossings_query(box, timeout, specific)

    r = httpx.post(OVERPASS_API_INTERPRETER, data={'data': query}, headers=http_headers(), timeout=timeout * 2)
    r.raise_for_status()

    elements = r.json()['elements']
    _extract_center(elements)

    result = []

    for e in elements:
        result.append(QueriedCrossing(
            position=LatLon(e['lat'], e['lon']),
            tags=e.get('tags', {}),
            bicycle=_is_bicycle(e)
        ))

    return tuple(result)


@retry(wait=wait_exponential(), stop=stop_after_delay(RETRY_TIME_LIMIT))
def query_buildings_roads(box: Box, *, interpolate_roads: bool = True) -> tuple[Sequence[LatLon], Sequence[LatLon]]:
    timeout = 180
    query = _build_buildings_roads_query(box, timeout)

    r = httpx.post(OVERPASS_API_INTERPRETER, data={'data': query}, headers=http_headers(), timeout=timeout * 2)
    r.raise_for_status()

    elements = r.json()['elements']
    _extract_center(elements)

    parts = _split_by_count(elements)
    buildings_elements = parts[0]
    roads_elements = parts[1]
    roads_nodes_elements = parts[2]

    buildings = tuple(
        LatLon(e['lat'], e['lon'])
        for e in buildings_elements
    )

    roads_nodes_position_map = {
        e['id']: LatLon(e['lat'], e['lon'])
        for e in roads_nodes_elements
    }

    roads = []

    for road_element in roads_elements:
        if not _is_road(road_element):
            continue

        for node_id in road_element['nodes']:
            roads.append(roads_nodes_position_map[node_id])

        if interpolate_roads:
            for n1_id, n2_id in pairwise(road_element['nodes']):
                n1_pos = roads_nodes_position_map[n1_id]
                n2_pos = roads_nodes_position_map[n2_id]

                distance = haversine_distance(n1_pos, n2_pos)
                num_interpolated = int(distance / GRID_FILTER_ROAD_INTERPOLATE)

                for i in range(1, num_interpolated + 1):
                    ratio = i / (num_interpolated + 1)
                    roads.append(LatLon(
                        n1_pos.lat + (n2_pos.lat - n1_pos.lat) * ratio,
                        n1_pos.lon + (n2_pos.lon - n1_pos.lon) * ratio,
                    ))

    return buildings, tuple(roads)


@retry(wait=wait_exponential(), stop=stop_after_delay(RETRY_TIME_LIMIT))
def query_roads_and_crossings_historical(boxes: Sequence[Box], max_age: float) -> Sequence[Sequence[QueriedRoadsAndCrossings]]:
    result = tuple([] for _ in boxes)

    for years_ago in (0, 0.3, 1, 2):
        result_historical = tuple(QueriedRoadsAndCrossings([], [], [], {}) for _ in boxes)

        timeout = 180
        query = _build_roads_query(boxes, timeout)

        if years_ago > 0:
            date = datetime.utcnow() - timedelta(days=365 * years_ago)
            date_fmt = date.strftime('%Y-%m-%dT%H:%M:%SZ')
            query = f'[date:"{date_fmt}"]{query}'

        r = httpx.post(OVERPASS_API_INTERPRETER, data={'data': query}, headers=http_headers(), timeout=timeout * 2)
        r.raise_for_status()

        data = r.json()
        data_timestamp = datetime \
            .strptime(data['osm3s']['timestamp_osm_base'], '%Y-%m-%dT%H:%M:%SZ') \
            .replace(tzinfo=UTC) \
            .timestamp()

        data_age = time() - data_timestamp
        if data_age > max_age:
            raise Exception(f'Overpass data is too old: {data_age} > {max_age}')

        elements = data['elements']
        _extract_center(elements)

        parts = _split_by_count(elements)
        assert len(parts) == len(boxes), f'Expected {len(boxes)} parts, got {len(parts)}'

        for i, slice in enumerate(parts):
            for e in slice:
                if e['type'] == 'way':
                    if _is_road(e):
                        result_historical[i].roads.append(e)
                    if _is_path(e):
                        result_historical[i].paths.append(e)
                elif e['type'] == 'node':
                    if _is_crossing(e):
                        result_historical[i].crossings.append(e)
                    result_historical[i].nodes[e['id']] = LatLon(e['lat'], e['lon'])

        for i, r in enumerate(result_historical):
            result[i].append(r)

    return result
