from datetime import datetime, timedelta
from typing import Iterable, NamedTuple, Sequence

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from box import Box
from config import OVERPASS_API_INTERPRETER
from latlon import LatLon
from polygon import Polygon
from utils import http_headers


class QueriedCrossing(NamedTuple):
    position: LatLon
    bicycle: bool


def _build_crossings_query(box: Box, timeout: int) -> str:
    return (
        f'[out:json][timeout:{timeout}][bbox:{box}];'
        f"nw[highway=crossing][crossing~'^(uncontrolled|marked|traffic_signals)$'];"
        f'out body center qt;'
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


def _is_bicycle(element: dict) -> bool:
    tags = element.get('tags', {})

    return (
        tags.get('bicycle', 'no') != 'no' or
        tags.get('crossing:markings', '') == 'dots'
    )


@retry(wait=wait_exponential(), stop=stop_after_attempt(5))
def query_crossings(box: Box, *, historical: bool) -> Sequence[QueriedCrossing]:
    result = []

    for years_ago in (0, 1, 2) if historical else (0,):
        timeout = 90
        query = _build_crossings_query(box, timeout)

        if years_ago > 0:
            date = datetime.utcnow() - timedelta(days=365 * years_ago)
            date_fmt = date.strftime('%Y-%m-%dT%H:%M:%SZ')
            query = f'[date:"{date_fmt}"]{query}'

        r = httpx.post(OVERPASS_API_INTERPRETER, data={'data': query}, headers=http_headers(), timeout=timeout * 2)
        r.raise_for_status()

        elements = r.json()['elements']

        for e in elements:
            if 'center' in e:
                e |= e['center']

            result.append(QueriedCrossing(
                position=LatLon(e['lat'], e['lon']),
                bicycle=_is_bicycle(e)
            ))

    return result
