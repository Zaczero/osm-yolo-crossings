import json
import pickle
import random
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import xmltodict
from skimage import img_as_float
from skimage.io import imread

from box import Box
from config import (CACHE_DIR, IMAGES_DIR, YOLO_DATASET_DIR,
                    YOLO_MODEL_RESOLUTION)
from db_grid import _GRID_SIZE_Y
from latlon import LatLon
from orto import fetch_orto
from overpass import query_elements_position
from polygon import Polygon
from processor import normalize_yolo_image
from utils import lat_to_meters, save_image


class YoloDatasetLabel(NamedTuple):
    polygons: Sequence[Polygon]
    labels: Sequence[int]


class YoloDatasetEntry(NamedTuple):
    id: str
    labels: YoloDatasetLabel
    image: np.ndarray


def _tag_to_label(tag: dict) -> int | None:
    if tag['@label'] in {'crossing'}:
        return 0

    raise ValueError(f'Unknown tag label: {tag["@label"]!r}')


def _iter_dataset_identifier(identifier: str, raw_path: Path, annotation: dict) -> YoloDatasetEntry | None:
    cache_path = CACHE_DIR / f'DatasetEntry_{identifier}.pkl'
    if cache_path.is_file():
        return pickle.loads(cache_path.read_bytes())

    # ignore images with labels
    if 'polygon' not in annotation:
        return None

    image = imread(raw_path)
    image = img_as_float(image)
    image = normalize_yolo_image(image)

    polygons = []
    labels = []

    for p in annotation['polygon']:
        label = _tag_to_label(p)
        if label is None:
            continue

        polygons.append(Polygon.from_str(p['@points']))
        labels.append(label)

    entry = YoloDatasetEntry(identifier, YoloDatasetLabel(polygons, labels), image)
    cache_path.write_bytes(pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))
    return entry


def iter_yolo_dataset() -> Iterable[YoloDatasetEntry]:
    cvat_annotation_paths = tuple(sorted(YOLO_DATASET_DIR.rglob('annotations.xml')))
    done = set()

    for i, p in enumerate(cvat_annotation_paths, 1):
        dir_progress = f'{i}/{len(cvat_annotation_paths)}'
        cvat_dir = p.parent
        print(f'[DATASET][{dir_progress}] 📁 Iterating: {cvat_dir!r}')

        cvat_annotations = xmltodict.parse(p.read_text(), force_list=('image', 'attribute', 'polygon'))
        cvat_annotations = cvat_annotations['annotations']['image']

        for j, annotation in enumerate(cvat_annotations, 1):
            file_progress = f'{j}/{len(cvat_annotations)}'
            raw_path = cvat_dir / Path(annotation['@name'])
            identifier = raw_path.stem
            if identifier in done:
                continue

            # print(f'[DATASET][{dir_progress}][{file_progress}] 📄 Iterating: {identifier!r}')

            entry = _iter_dataset_identifier(identifier, raw_path, annotation)
            if entry is None:
                continue

            done.add(identifier)
            yield entry


class ProcessCellResult(NamedTuple):
    box: Box
    image: np.ndarray
    overlay: np.ndarray


def _process(size: int) -> Sequence[ProcessCellResult]:
    crossings = query_elements_position('node[highway=crossing][!bicycle]')
    crossings = list(crossings)
    random.shuffle(crossings)

    if crossings:
        print(f'[DATASET] 🦓 Processing {len(crossings)} elements')

    result = []

    for crossing_position in crossings:
        crossing_box = Box(crossing_position, LatLon(0, 0))
        crossing_box = crossing_box.extend(meters=lat_to_meters(_GRID_SIZE_Y) / 2)

        orto_img = fetch_orto(crossing_box, YOLO_MODEL_RESOLUTION)
        if orto_img is None:
            continue

        overlay_img = orto_img.copy()
        overlay_img = normalize_yolo_image(overlay_img)

        save_image(orto_img, 'dataset_yolo_1')
        save_image(overlay_img, 'dataset_yolo_2')

        result.append(ProcessCellResult(
            box=crossing_box,
            image=orto_img,
            overlay=overlay_img,
        ))

        if len(result) >= size:
            break

    return tuple(result)


def create_yolo_dataset(size: int) -> None:
    cvat_annotations = {
        'annotations': {
            'version': '1.1',
            'image': []
        }
    }

    cvat_image_annotations: list[dict] = cvat_annotations['annotations']['image']

    for result in _process(size):
        unique_id = str(result.box).replace('.', '_')

        raw_path = save_image(result.image, f'CVAT/images/{unique_id}', force=True)
        raw_name = raw_path.name
        raw_name_safe = raw_name.replace('.', '_')

        save_image(result.overlay, f'CVAT/images/related_images/{raw_name_safe}/overlay', force=True)

        with open(IMAGES_DIR / f'CVAT/images/related_images/{raw_name_safe}/box.json', 'w') as f:
            json.dump(result.box, f)

        annotation = {
            '@name': f'images/{raw_name}',
            '@height': result.image.shape[0],
            '@width': result.image.shape[1],
        }

        cvat_image_annotations.append(annotation)

    # sort in lexical order
    cvat_image_annotations.sort(key=lambda x: x['@name'])

    # add numerical ids
    for i, annotation in enumerate(cvat_image_annotations):
        annotation['@id'] = i

    with open(IMAGES_DIR / 'CVAT/annotations.xml', 'w') as f:
        xmltodict.unparse(cvat_annotations, output=f, pretty=True)
