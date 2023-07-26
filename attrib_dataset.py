import json
import pickle
import traceback
from functools import partial
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from re import I
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import xmltodict
from skimage import draw, img_as_float, transform
from skimage.io import imread

from box import Box
from config import (ATTRIB_DATASET_DIR, ATTRIB_MODEL_PATH,
                    ATTRIB_MODEL_RESOLUTION, CACHE_DIR, CPU_COUNT, IMAGES_DIR,
                    YOLO_MODEL_RESOLUTION)
from db_grid import random_grid
from latlon import LatLon
from orto import FetchMode, fetch_orto
from overpass import query_specific_crossings
from polygon2 import Polygon2
from processor import (ProcessPolygonResult, normalize_attrib_image,
                       normalize_yolo_image, process_image, process_polygon)
from transform_geo_px import transform_rad_to_px
from utils import print_run_time, save_image

_EXTEND = 9


class AttribDatasetLabel(NamedTuple):
    labels: Sequence[int]


class AttribDatasetEntry(NamedTuple):
    id: str
    labels: AttribDatasetLabel
    image: np.ndarray


def _tag_to_label(tag: dict) -> int | None:
    if tag['@label'] in {'invalid'}:
        return None
    if tag['@label'] in {'valid'}:
        return 0
    if tag['@label'] in {'signals'}:
        return 1
    if tag['@label'] in {'raised'}:
        return 2

    raise ValueError(f'Unknown tag label: {tag["@label"]!r}')


def _iter_dataset_identifier(identifier: str, raw_path: Path, annotation: dict) -> AttribDatasetEntry | None:
    cache_path = CACHE_DIR / f'AttribDatasetEntry_{identifier}.pkl'
    if cache_path.is_file():
        return pickle.loads(cache_path.read_bytes())

    if 'tag' not in annotation:
        return None

    image = imread(raw_path)
    image = img_as_float(image)
    image = normalize_attrib_image(image)

    labels = []

    for p in annotation['tag']:
        label = _tag_to_label(p)
        if label is None:
            continue

        labels.append(label)

    entry = AttribDatasetEntry(identifier, AttribDatasetLabel(labels), image)
    cache_path.write_bytes(pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))
    return entry


def iter_attrib_dataset() -> Iterable[AttribDatasetEntry]:
    cvat_annotation_paths = tuple(sorted(ATTRIB_DATASET_DIR.rglob('annotations.xml')))
    done = set()

    for i, p in enumerate(cvat_annotation_paths, 1):
        dir_progress = f'{i}/{len(cvat_annotation_paths)}'
        cvat_dir = p.parent
        print(f'[DATASET][{dir_progress}] 📁 Iterating: {cvat_dir!r}')

        cvat_annotations = xmltodict.parse(p.read_text(), force_list=('image', 'attribute', 'tag'))
        cvat_annotations = cvat_annotations['annotations']['image']

        for j, annotation in enumerate(cvat_annotations, 1):
            file_progress = f'{j}/{len(cvat_annotations)}'
            raw_path = cvat_dir / Path(annotation['@name'])
            identifier = raw_path.stem
            if identifier in done:
                continue

            print(f'[DATASET][{dir_progress}][{file_progress}] 📄 Iterating: {identifier!r}')

            entry = _iter_dataset_identifier(identifier, raw_path, annotation)
            if entry is None:
                continue

            done.add(identifier)
            yield entry


class ProcessCellResult(NamedTuple):
    position: LatLon
    image: np.ndarray
    overlay: np.ndarray
    attributes: Sequence[str]


def _process_cell(cell: Box) -> Sequence[ProcessCellResult]:
    crossings = query_specific_crossings(cell, "~'^(uncontrolled|marked|traffic_signals)$'", historical=False)

    if crossings:
        print(f'[DATASET] 🦓 Processing {len(crossings)} crossings')

    result = []

    for crossing in crossings:
        crossing_box = Box(crossing.position, LatLon(0, 0))
        crossing_box = crossing_box.extend(meters=_EXTEND)

        orto_img = fetch_orto(crossing_box, FetchMode.FAST, resolution=ATTRIB_MODEL_RESOLUTION)

        if orto_img is None:
            continue

        overlay_img = orto_img.copy()
        overlay_img = normalize_attrib_image(overlay_img)

        crossing_px = transform_rad_to_px(
            (crossing.position,),
            img_box=crossing_box,
            img_shape=overlay_img.shape)[0]

        rr, cc = draw.disk(crossing_px, radius=5, shape=overlay_img.shape[:2])
        overlay_img[rr, cc] = (0, 0, 0)
        rr, cc = draw.disk(crossing_px, radius=4, shape=overlay_img.shape[:2])
        overlay_img[rr, cc] = (1, 0, 0)

        save_image(orto_img, 'dataset_attrib_1')
        save_image(overlay_img, 'dataset_attrib_2')

        attributes = []

        if crossing.tags.get('crossing', '') == 'traffic_signals':
            attributes.append('signals')

        if crossing.tags.get('traffic_calming', '') == 'table':
            attributes.append('raised')

        result.append(ProcessCellResult(
            position=crossing.position,
            image=orto_img,
            overlay=overlay_img,
            attributes=tuple(attributes),
        ))

    return tuple(result)


def create_attrib_dataset(size: int) -> None:
    cvat_annotations = {
        'annotations': {
            'version': '1.1',
            'image': []
        }
    }

    cvat_image_annotations: list[dict] = cvat_annotations['annotations']['image']

    with Pool(CPU_COUNT) as pool:
        cells = random_grid()
        for i in range(0, len(cells), CPU_COUNT):
            process_cells = cells[i:i+CPU_COUNT]

            if CPU_COUNT == 1:
                iterator = map(_process_cell, process_cells)
            else:
                iterator = pool.imap_unordered(_process_cell, process_cells)

            for result in chain.from_iterable(iterator):
                unique_id = str(result.position).replace('.', '_')
                category = '_'.join(sorted(result.attributes)) or 'generic'

                raw_path = save_image(result.image, f'CVAT/{category}/{unique_id}', force=True)
                raw_name = raw_path.name
                raw_name_safe = raw_name.replace('.', '_')

                save_image(result.overlay, f'CVAT/{category}/related_images/{raw_name_safe}/overlay', force=True)

                with open(IMAGES_DIR / f'CVAT/{category}/related_images/{raw_name_safe}/position.json', 'w') as f:
                    json.dump(result.position, f)

                annotation = {
                    '@name': f'{category}/{raw_name}',
                    '@height': result.image.shape[0],
                    '@width': result.image.shape[1],
                }

                cvat_image_annotations.append(annotation)

            if len(cvat_image_annotations) >= size:
                break

    # sort in lexical order
    cvat_image_annotations.sort(key=lambda x: x['@name'])

    # add numerical ids
    for i, annotation in enumerate(cvat_image_annotations):
        annotation['@id'] = i

    with open(IMAGES_DIR / 'CVAT/annotations.xml', 'w') as f:
        xmltodict.unparse(cvat_annotations, output=f, pretty=True)
