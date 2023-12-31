from typing import Sequence

import httpx
import xmltodict
from cachetools import TTLCache, cached
from tenacity import (retry, stop_after_attempt, stop_after_delay,
                      wait_exponential)

from config import (CHANGESET_ID_PLACEHOLDER, DEFAULT_CHANGESET_TAGS,
                    OSM_PASSWORD, OSM_USERNAME, RETRY_TIME_LIMIT)
from utils import http_headers
from xmltodict_postprocessor import xmltodict_postprocessor


class OpenStreetMap:
    def _get_http_client(self) -> httpx.Client:
        return httpx.Client(base_url='https://api.openstreetmap.org/api',
                            headers=http_headers(),
                            auth=(OSM_USERNAME, OSM_PASSWORD),
                            timeout=60)

    @retry(wait=wait_exponential(), stop=stop_after_attempt(3))
    def get_changeset_max_size(self) -> int:
        with self._get_http_client() as http:
            r = http.get('/capabilities')
            r.raise_for_status()

        caps = xmltodict.parse(r.text)

        return int(caps['osm']['api']['changesets']['@maximum_elements'])

    def get_relation(self, relation_id: str | int,) -> dict:
        return (self._get_elements('relations', (relation_id,)))[0]

    def get_way(self, way_id: str | int,) -> dict:
        return (self._get_elements('ways', (way_id,)))[0]

    def get_node(self, node_id: str | int,) -> dict:
        return (self._get_elements('nodes', (node_id,)))[0]

    def get_relations(self, relation_ids: Sequence[str | int],) -> list[dict]:
        return self._get_elements('relations', relation_ids)

    def get_ways(self, way_ids: Sequence[str | int],) -> list[dict]:
        return self._get_elements('ways', way_ids)

    def get_nodes(self, node_ids: Sequence[str | int],) -> list[dict]:
        return self._get_elements('nodes', node_ids)

    @cached(TTLCache(1024, ttl=60))
    @retry(wait=wait_exponential(max=1800), stop=stop_after_delay(RETRY_TIME_LIMIT))
    def _get_elements(self, elements_type: str, element_ids: Sequence[str]) -> list[dict]:
        batch_size = 500
        results = []

        if not element_ids:
            return results

        def _get_batch(batch: Sequence[str]) -> list[dict]:
            with self._get_http_client() as http:
                r = http.get(f'/0.6/{elements_type}', params={elements_type: ','.join(map(str, batch))})
                r.raise_for_status()

            data = xmltodict.parse(
                r.text,
                postprocessor=xmltodict_postprocessor,
                force_list=('node', 'way', 'relation', 'member', 'tag', 'nd'),
            )['osm']

            return data[elements_type[:-1]]

        for i in range(0, len(element_ids), batch_size):
            batch = element_ids[i:i + batch_size]
            results.extend(_get_batch(batch))

        return results

    @retry(wait=wait_exponential(max=1800), stop=stop_after_delay(RETRY_TIME_LIMIT))
    def get_way_full(self, way_id: str | int,) -> dict:
        with self._get_http_client() as http:
            r = http.get(f'/0.6/way/{way_id}/full')
            r.raise_for_status()

        data = xmltodict.parse(
            r.text,
            postprocessor=xmltodict_postprocessor,
            force_list=('node', 'way', 'relation', 'member', 'tag', 'nd'),
        )['osm']

        return data

    @retry(wait=wait_exponential(), stop=stop_after_attempt(3))
    def get_authorized_user(self) -> dict | None:
        with self._get_http_client() as http:
            r = http.get('/0.6/user/details.json')
            r.raise_for_status()

        return r.json()['user']

    @retry(wait=wait_exponential(), stop=stop_after_attempt(2))
    def upload_osm_change(self, osm_change: str) -> str:
        changeset = xmltodict.unparse({'osm': {'changeset': {'tag': [
            {'@k': k, '@v': v}
            for k, v in DEFAULT_CHANGESET_TAGS.items()
        ]}}})

        with self._get_http_client() as http:
            r = http.put('/0.6/changeset/create', content=changeset, headers={
                'Content-Type': 'text/xml; charset=utf-8'}, follow_redirects=False)
            r.raise_for_status()

            changeset_id = r.text
            osm_change = osm_change.replace(CHANGESET_ID_PLACEHOLDER, changeset_id)
            print(f'🌐 Changeset: https://www.openstreetmap.org/changeset/{changeset_id}')

            upload_resp = http.post(f'/0.6/changeset/{changeset_id}/upload', content=osm_change, headers={
                'Content-Type': 'text/xml; charset=utf-8'}, timeout=150)

            r = http.put(f'/0.6/changeset/{changeset_id}/close')
            r.raise_for_status()

        if not upload_resp.is_success:
            raise Exception(f'Upload failed ({upload_resp.status_code}): {upload_resp.text}')

        return changeset_id
