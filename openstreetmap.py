import httpx
import xmltodict
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (CHANGESET_ID_PLACEHOLDER, DEFAULT_CHANGESET_TAGS,
                    OSM_PASSWORD, OSM_USERNAME)
from utils import http_headers


class OpenStreetMap:
    def _get_http_client(self) -> httpx.Client:
        return httpx.Client(base_url='https://api.openstreetmap.org/api',
                            headers=http_headers(),
                            auth=(OSM_USERNAME, OSM_PASSWORD))

    def get_changeset_max_size(self) -> int:
        with self._get_http_client() as http:
            r = http.get('/capabilities')
            r.raise_for_status()

        caps = xmltodict.parse(r.text)

        return int(caps['osm']['api']['changesets']['@maximum_elements'])

    @retry(wait=wait_exponential(), stop=stop_after_attempt(3))
    def get_authorized_user(self) -> dict | None:
        with self._get_http_client() as http:
            r = http.get('/0.6/user/details.json')
            r.raise_for_status()

        return r.json()['user']

    @retry(wait=wait_exponential(), stop=stop_after_attempt(3))
    def upload_osm_change(self, osm_change: str, comment_extra: str) -> str:
        changeset = xmltodict.unparse({'osm': {'changeset': {'tag': [
            {'@k': k, '@v': f'{v}: {comment_extra}'}
            if k == 'comment' else
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
