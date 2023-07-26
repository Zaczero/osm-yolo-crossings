from datetime import datetime, timedelta
from typing import Iterable, NamedTuple, Sequence

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from box import Box
from config import OVERPASS_API_INTERPRETER, SEARCH_RELATION
from latlon import LatLon
from polygon import Polygon
from utils import http_headers


class QueriedCrossing(NamedTuple):
    position: LatLon
    tags: dict[str, str]
    bicycle: bool


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

    return tags.get('highway', '') in {
        'residential',
        'service',
        'unclassified',
        'tertiary',
        'secondary',
        'primary',
        'living_street',
        'road',
    }


@retry(wait=wait_exponential(), stop=stop_after_attempt(5))
def query_specific_crossings(box: Box, specific: str, *, historical: bool) -> Sequence[QueriedCrossing]:
    result = []

    for years_ago in (0, 1, 2) if historical else (0,):
        timeout = 90
        query = _build_specific_crossings_query(box, timeout, specific)

        if years_ago > 0:
            date = datetime.utcnow() - timedelta(days=365 * years_ago)
            date_fmt = date.strftime('%Y-%m-%dT%H:%M:%SZ')
            query = f'[date:"{date_fmt}"]{query}'

        r = httpx.post(OVERPASS_API_INTERPRETER, data={'data': query}, headers=http_headers(), timeout=timeout * 2)
        r.raise_for_status()

        elements = r.json()['elements']
        _extract_center(elements)

        for e in elements:
            result.append(QueriedCrossing(
                position=LatLon(e['lat'], e['lon']),
                tags=e.get('tags', {}),
                bicycle=_is_bicycle(e)
            ))

    return result


@retry(wait=wait_exponential(), stop=stop_after_attempt(5))
def query_buildings_roads(box: Box) -> tuple[Sequence[LatLon], Sequence[LatLon]]:
    timeout = 90
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

    roads_nodes_map = {
        e['id']: LatLon(e['lat'], e['lon'])
        for e in roads_nodes_elements
    }

    roads = []

    for road_element in roads_elements:
        if not _is_road(road_element):
            continue

        for node_id in road_element['nodes']:
            roads.append(roads_nodes_map[node_id])

    return buildings, tuple(roads)
