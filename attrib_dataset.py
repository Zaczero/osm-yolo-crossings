import json
import pickle
import random
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import xmltodict
from skimage import draw, img_as_float
from skimage.io import imread
from sklearn.preprocessing import MultiLabelBinarizer

from box import Box
from config import (ATTRIB_DATASET_DIR, ATTRIB_MODEL_RESOLUTION,
                    ATTRIB_NUM_CLASSES, ATTRIB_POSITION_EXTEND, CACHE_DIR,
                    CPU_COUNT, IMAGES_DIR)
from db_grid import random_grid
from latlon import LatLon
from orto import fetch_orto
from overpass import query_elements_position, query_specific_crossings
from processor import normalize_attrib_image
from transform_geo_px import transform_rad_to_px
from utils import save_image


class AttribDatasetLabel(NamedTuple):
    labels: Sequence[int]

    @property
    def is_valid(self) -> bool:
        return 0 in self.labels

    # def encode(self) -> np.ndarray:
    #     return MultiLabelBinarizer(classes=range(ATTRIB_NUM_CLASSES)).fit_transform([self.labels])[0]

    # def encode_num(self) -> int:
    #     return int(''.join(map(str, self.encode())), 2)


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
        return None  # too few samples

    raise ValueError(f'Unknown tag label: {tag["@label"]!r}')


def _iter_dataset_identifier(identifier: str, raw_path: Path, annotation: dict) -> AttribDatasetEntry | None:
    cache_path = CACHE_DIR / f'AttribDatasetEntry_{identifier}.pkl'
    if cache_path.is_file():
        return pickle.loads(cache_path.read_bytes())

    if 'tag' not in annotation:
        return None

    labels = []

    for p in annotation['tag']:
        label = _tag_to_label(p)
        if label is None:
            continue

        labels.append(label)

    image = imread(raw_path)
    image = img_as_float(image)
    image = normalize_attrib_image(image)

    # center_x = image.shape[1] / 2
    # center_y = image.shape[0] / 2
    # center = (center_y, center_x)

    # rr, cc = draw.disk(center, radius=6, shape=image.shape[:2])
    # image[rr, cc] = (0, 0, 0)
    # rr, cc = draw.disk(center, radius=4, shape=image.shape[:2])
    # image[rr, cc] = (1, 1, 1)

    save_image(image, 'dataset_attrib_1')

    image = image * 2 - 1  # MobileNet requires [-1, 1] input

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

            # print(f'[DATASET][{dir_progress}][{file_progress}] 📄 Iterating: {identifier!r}')

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


def _process(size: int) -> Sequence[ProcessCellResult]:
    attributes = ('signals',)

    crossings = query_elements_position('node[crossing=traffic_signals][!bicycle]')
    crossings = list(crossings)
    random.shuffle(crossings)

    if crossings:
        print(f'[DATASET] 🦓 Processing {len(crossings)} elements')

    result = []

    for crossing_position in crossings:
        crossing_box = Box(crossing_position, LatLon(0, 0))
        crossing_box = crossing_box.extend(meters=ATTRIB_POSITION_EXTEND)

        orto_img = fetch_orto(crossing_box, ATTRIB_MODEL_RESOLUTION)
        if orto_img is None:
            continue

        overlay_img = orto_img.copy()
        overlay_img = normalize_attrib_image(overlay_img)

        crossing_px = transform_rad_to_px(
            (crossing_position,),
            img_box=crossing_box,
            img_shape=overlay_img.shape)[0]

        rr, cc = draw.disk(crossing_px, radius=5, shape=overlay_img.shape[:2])
        overlay_img[rr, cc] = (0, 0, 0)
        rr, cc = draw.disk(crossing_px, radius=4, shape=overlay_img.shape[:2])
        overlay_img[rr, cc] = (1, 0, 0)

        save_image(orto_img, 'dataset_attrib_1')
        save_image(overlay_img, 'dataset_attrib_2')

        # if crossing.tags.get('crossing', '') == 'traffic_signals':
        #     attributes.append('signals')

        # if crossing.tags.get('traffic_calming', '') == 'table':
        #     attributes.append('raised')

        result.append(ProcessCellResult(
            position=crossing_position,
            image=orto_img,
            overlay=overlay_img,
            attributes=tuple(attributes),
        ))

        if len(result) >= size:
            break

    return tuple(result)


def create_attrib_dataset(size: int) -> None:
    cvat_annotations = {
        'annotations': {
            'version': '1.1',
            'image': []
        }
    }

    cvat_image_annotations: list[dict] = cvat_annotations['annotations']['image']

    for result in _process(size):
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

    # sort in lexical order
    cvat_image_annotations.sort(key=lambda x: x['@name'])

    # add numerical ids
    for i, annotation in enumerate(cvat_image_annotations):
        annotation['@id'] = i

    with open(IMAGES_DIR / 'CVAT/annotations.xml', 'w') as f:
        xmltodict.unparse(cvat_annotations, output=f, pretty=True)
