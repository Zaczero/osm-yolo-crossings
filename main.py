import random
import traceback
from multiprocessing import Pool
from time import sleep
from typing import Sequence

import numpy as np

from attrib_dataset import create_attrib_dataset
from attrib_model import create_attrib_model
from box import Box
# from check_on_osm import check_on_osm
from config import (CPU_COUNT, SEED, SLEEP_AFTER_ONE_IMPORT, YOLO_CONFIDENCE,
                    YOLO_MODEL_RESOLUTION)
# from db_added import filter_added, mark_added
from db_grid import iter_grid
from latlon import LatLon
from openstreetmap import OpenStreetMap
from orto import FetchMode, fetch_orto
# from osm_change import create_buildings_change
from processor import normalize_yolo_image
from transform_geo_px import transform_px_to_rad
# from processor import process_image, process_polygon
# from tuned_model import TunedModel
from utils import index_box_centered, print_run_time, save_image
from yolo_dataset import create_yolo_dataset
from yolo_model import create_yolo_model
from yolo_tuned_model import YoloTunedModel

random.seed(SEED)
np.random.seed(SEED)


_MIN_EDGE_DISTANCE = 0.1


def _process_orto(model: YoloTunedModel, cell: Box, orto_img: np.ndarray) -> Sequence[Box]:
    assert orto_img.shape[0] == orto_img.shape[1]
    num_subcells = int(orto_img.shape[0] / YOLO_MODEL_RESOLUTION)
    subcell_size_lat = cell.size.lat / num_subcells
    subcell_size_lon = cell.size.lon / num_subcells
    result = []

    for y in range(num_subcells):
        subcell_lat = cell.point.lat + cell.size.lat - subcell_size_lat * (y + 1)
        for x in range(num_subcells):
            subcell_lon = cell.point.lon + x * subcell_size_lon

            subcell = Box(LatLon(subcell_lat, subcell_lon), LatLon(subcell_size_lat, subcell_size_lon))
            subcell_img = orto_img[
                y * YOLO_MODEL_RESOLUTION:(y + 1) * YOLO_MODEL_RESOLUTION,
                x * YOLO_MODEL_RESOLUTION:(x + 1) * YOLO_MODEL_RESOLUTION,
                :
            ]

            save_image(subcell_img, f'subcell_{y}_{x}')
            pred = model.predict_single(subcell_img)

            boxes = pred['boxes']
            confidences = pred['confidence']
            classes = pred['classes']

            if boxes:
                print(f'[PROCESS] 💡 Found {len(boxes)} detections')

            for box, confidence, _ in zip(boxes, confidences, classes):
                if confidence < YOLO_CONFIDENCE:
                    continue

                box_cell = subcell
                max_repeat = 2

                for r in range(max_repeat):
                    x_min, y_min, w, h = box
                    x_max = x_min + w
                    y_max = y_min + h

                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2

                    x_min_p = x_min / YOLO_MODEL_RESOLUTION
                    y_min_p = y_min / YOLO_MODEL_RESOLUTION
                    x_max_p = x_max / YOLO_MODEL_RESOLUTION
                    y_max_p = y_max / YOLO_MODEL_RESOLUTION

                    if r + 1 == max_repeat or (
                        _MIN_EDGE_DISTANCE < x_min_p < 1 - _MIN_EDGE_DISTANCE and
                        _MIN_EDGE_DISTANCE < x_max_p < 1 - _MIN_EDGE_DISTANCE and
                        _MIN_EDGE_DISTANCE < y_min_p < 1 - _MIN_EDGE_DISTANCE and
                        _MIN_EDGE_DISTANCE < y_max_p < 1 - _MIN_EDGE_DISTANCE
                    ):
                        points = transform_px_to_rad(
                            ((x_min, y_max), (x_max, y_min)), box_cell,
                            (YOLO_MODEL_RESOLUTION, YOLO_MODEL_RESOLUTION))[0]
                        p1 = LatLon(points[0][0], points[0][1])
                        p2 = LatLon(points[1][0], points[1][1])
                        result.append(Box(p1, p2 - p1))
                        break

                    tx = center_x - YOLO_MODEL_RESOLUTION / 2
                    ty = center_y - YOLO_MODEL_RESOLUTION / 2

                    tx_p = tx / YOLO_MODEL_RESOLUTION
                    ty_p = ty / YOLO_MODEL_RESOLUTION

                    box_cell_translate = LatLon(
                        -ty_p * subcell.size.lat,
                        tx_p * subcell.size.lon)

                    box_cell = Box(box_cell.point + box_cell_translate, box_cell.size)

                    box_img = fetch_orto(box_cell, FetchMode.FAST, YOLO_MODEL_RESOLUTION)
                    box_img = normalize_yolo_image(box_img)

                    pred = model.predict_single(box_img)
                    boxes, confidences = pred['boxes'], pred['confidence']

                    if not boxes:
                        break

                    centered_i = index_box_centered(boxes, YOLO_MODEL_RESOLUTION)
                    box, confidence = boxes[centered_i], confidences[centered_i]

    return result


def main() -> None:
    with print_run_time('Logging in'):
        osm = OpenStreetMap()
        display_name = osm.get_authorized_user()['display_name']
        print(f'👤 Welcome, {display_name}!')

        changeset_max_size = osm.get_changeset_max_size()

    with print_run_time('Loading YOLO model'):
        yolo_model = YoloTunedModel()

    with Pool(CPU_COUNT) as pool:
        for cell in iter_grid():
            print(f'[CELL] ⚙️ Processing {cell!r}')

            with print_run_time('Fetching ortophoto imagery'):
                orto_img = fetch_orto(cell, FetchMode.FAST)

            if orto_img is None:
                print('[CELL] ⏭️ Nothing to do: missing ortophoto')
                continue

            orto_img = normalize_yolo_image(orto_img)
            crossings = _process_orto(yolo_model, cell, orto_img)

            # valid_buildings: list[ClassifiedBuilding] = []

            # if CPU_COUNT == 1:
            #     iterator = map(_process_building, buildings)
            # else:
            #     iterator = pool.imap_unordered(_process_building, buildings, chunksize=4)

            # for building, model_input in iterator:
            #     if model_input is None:
            #         print(f'[PROCESS] 🚫 Unsupported')
            #         mark_added((building,), reason='unsupported')
            #         continue

            #     is_valid, proba = model.predict_single(model_input)

            #     if is_valid:
            #         print(f'[PROCESS][{proba:.3f}] ✅ Valid')
            #         valid_buildings.append(ClassifiedBuilding(building, proba))
            #     else:
            #         print(f'[PROCESS][{proba:.3f}] 🚫 Invalid')
            #         mark_added((building,), reason='predict', predict=proba)

            # print(f'[CELL][1/2] 🏠 Valid buildings: {len(valid_buildings)}')

            # with print_run_time('Check on OSM'):
            #     found, not_found = check_on_osm(valid_buildings)
            #     assert len(found) + len(not_found) == len(valid_buildings)

            # if found:
            #     mark_added(tuple(map(lambda cb: cb.building, found)), reason='found')

            # print(f'[CELL][2/2] 🏠 Needing import: {len(not_found)}')

            # if not_found:
            #     for score_min, score_max, name in ((0.999, 1.001, '>99.9%',),
            #                                        (0.000, 0.999, '>99.5%',)):
            #         buildings = tuple(cb.building for cb in not_found if score_min <= cb.score < score_max)

            #         for chunk in building_chunks(buildings, size=changeset_max_size):
            #             with print_run_time('Create OSM change'):
            #                 osm_change = create_buildings_change(chunk)

            #             with print_run_time('Upload OSM change'):
            #                 if DRY_RUN:
            #                     print('[DRY-RUN] 🚫 Skipping upload')
            #                     changeset_id = 0
            #                 else:
            #                     changeset_id = osm.upload_osm_change(osm_change, name)

            #             mark_added(chunk, reason='upload', changeset_id=changeset_id)
            #             print(f'✅ Import successful: {name!r} ({len(chunk)})')

            #     if SLEEP_AFTER_ONE_IMPORT:
            #         sleep_duration = len(not_found) * SLEEP_AFTER_ONE_IMPORT
            #         print(f'[SLEEP-IMPORT] ⏳ Sleeping for {sleep_duration} seconds...')
            #         sleep(sleep_duration)


if __name__ == '__main__':
    # create_dataset(1000)
    # process_dataset()
    # create_model()
    # create_attrib_dataset(1000)
    create_yolo_model()
    # main()
