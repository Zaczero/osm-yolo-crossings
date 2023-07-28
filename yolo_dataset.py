import json
import pickle
import traceback
from functools import partial
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import xmltodict
from skimage import draw, img_as_float, transform
from skimage.io import imread

from box import Box
from config import (CACHE_DIR, CPU_COUNT, IMAGES_DIR, YOLO_DATASET_DIR,
                    YOLO_MODEL_RESOLUTION)
from db_grid import random_grid
from latlon import LatLon
from orto import FetchMode, fetch_orto
from overpass import query_specific_crossings
from polygon2 import Polygon2
from processor import (ProcessPolygonResult, normalize_yolo_image,
                       process_image, process_polygon)
from transform_geo_px import transform_rad_to_px
from utils import print_run_time, save_image
from yolo_tuned_model import YoloTunedModel


class YoloDatasetLabel(NamedTuple):
    polygons: Sequence[Polygon2]
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

        polygons.append(Polygon2.from_str(p['@points']))
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


def _process_cell(cell: Box, *, must_contain_crossings: bool) -> Sequence[ProcessCellResult]:
    crossings = query_specific_crossings(cell, "~'^(uncontrolled|marked|traffic_signals)$'")
    if len(crossings) < 2:
        return []

    orto_img = fetch_orto(cell, FetchMode.FAST)
    if orto_img is None:
        return []

    save_image(orto_img, '1')
    print(f'[DATASET] 🦓 Processing {len(crossings)} crossings')

    assert orto_img.shape[0] == orto_img.shape[1]
    num_subcells = int(orto_img.shape[0] / YOLO_MODEL_RESOLUTION)
    subcell_size_lat = cell.size.lat / num_subcells
    subcell_size_lon = cell.size.lon / num_subcells
    result = []

    for y in range(int(num_subcells)):
        subcell_lat = cell.point.lat + cell.size.lat - subcell_size_lat * (y + 1)
        for x in range(int(num_subcells)):
            subcell_lon = cell.point.lon + subcell_size_lon * x

            subcell = Box(LatLon(subcell_lat, subcell_lon), LatLon(subcell_size_lat, subcell_size_lon))
            subcell_crossings = tuple(c for c in crossings if c.position in subcell)

            if must_contain_crossings and not subcell_crossings:
                continue

            subcell_orto_img = orto_img[
                y * YOLO_MODEL_RESOLUTION:(y + 1) * YOLO_MODEL_RESOLUTION,
                x * YOLO_MODEL_RESOLUTION:(x + 1) * YOLO_MODEL_RESOLUTION,
                :
            ]

            subcell_crossings_px = transform_rad_to_px(
                (c.position for c in subcell_crossings),
                img_box=subcell,
                img_shape=subcell_orto_img.shape)

            subcell_overlay_img = subcell_orto_img.copy()
            subcell_overlay_img = normalize_yolo_image(subcell_overlay_img)

            for crossing, crossing_px in zip(subcell_crossings, subcell_crossings_px):
                rr, cc = draw.disk(crossing_px, radius=6, shape=subcell_orto_img.shape[:2])
                subcell_overlay_img[rr, cc] = (0, 0, 0)
                rr, cc = draw.disk(crossing_px, radius=5, shape=subcell_orto_img.shape[:2])
                subcell_overlay_img[rr, cc] = (1, 0, 1) if crossing.bicycle else (1, 0, 0)

            save_image(subcell_orto_img, 'dataset_yolo_1')
            save_image(subcell_overlay_img, 'dataset_yolo_2')

            result.append(ProcessCellResult(
                box=subcell,
                image=subcell_orto_img,
                overlay=subcell_overlay_img,
            ))

    return tuple(result)


def create_yolo_dataset(size: int) -> None:
    cvat_annotations = {
        'annotations': {
            'version': '1.1',
            'image': []
        }
    }

    cvat_image_annotations: list[dict] = cvat_annotations['annotations']['image']

    with Pool(CPU_COUNT) as pool:
        process_func = partial(_process_cell, must_contain_crossings=True)
        cells = random_grid()
        for i in range(0, len(cells), CPU_COUNT):
            process_cells = cells[i:i+CPU_COUNT]

            if CPU_COUNT == 1:
                iterator = map(process_func, process_cells)
            else:
                iterator = pool.imap_unordered(process_func, process_cells)

            for result in chain.from_iterable(iterator):
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

            if len(cvat_image_annotations) >= size:
                break

    # sort in lexical order
    cvat_image_annotations.sort(key=lambda x: x['@name'])

    # add numerical ids
    for i, annotation in enumerate(cvat_image_annotations):
        annotation['@id'] = i

    with open(IMAGES_DIR / 'CVAT/annotations.xml', 'w') as f:
        xmltodict.unparse(cvat_annotations, output=f, pretty=True)
