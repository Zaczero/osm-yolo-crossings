import asyncio
import json
import random
from functools import cache
from itertools import islice
from time import sleep
from typing import Sequence

import numpy as np
from skimage import draw
from sklearn.neighbors import BallTree

from attrib_dataset import create_attrib_dataset
from attrib_model import create_attrib_model
from attrib_tuned_model import AttribTunedModel
from box import Box
from config import (ADDED_SEARCH_RADIUS, ATTRIB_MODEL_RESOLUTION,
                    ATTRIB_POSITION_EXTEND, CPU_COUNT, CROSSING_BOX_EXTEND,
                    DATA_DIR, DRY_RUN, MIN_IMPORT_SIZE, SEED,
                    SLEEP_AFTER_GRID_ITER, YOLO_MODEL_RESOLUTION)
from crossing_merger import CrossingMergeInstructions, merge_crossings
from crossing_suggestion import CrossingSuggestion
from db_added import filter_not_added, mark_added
from db_grid import Cell, iter_grid, set_last_cell
from latlon import LatLon
from openstreetmap import OpenStreetMap
from orto import fetch_orto, fetch_orto_async
from osm_change import create_instructed_change
from processor import normalize_attrib_image, normalize_yolo_image
from transform_geo_px import transform_px_to_rad
from utils import (index_box_centered, meters_to_lat, print_run_time,
                   save_image, set_nice, sleep_after_import)
from yolo_dataset import create_yolo_dataset
from yolo_model import create_yolo_model
from yolo_tuned_model import YoloTunedModel

random.seed(SEED)
np.random.seed(SEED)


_PROCESS_NICE = 15
_MIN_EDGE_DISTANCE = 0.1


@cache
def _get_yolo_model() -> YoloTunedModel:
    with print_run_time('Loading YOLO model'):
        return YoloTunedModel()


@cache
def _get_attrib_model() -> AttribTunedModel:
    with print_run_time('Loading ATTRIB model'):
        return AttribTunedModel()


def _process_orto(cell_box: Box, orto_img: np.ndarray) -> Sequence[Box]:
    yolo_model = _get_yolo_model()

    assert orto_img.shape[0] == orto_img.shape[1]
    num_subcells = int(orto_img.shape[0] / YOLO_MODEL_RESOLUTION)
    subcell_size_lat = cell_box.size.lat / num_subcells
    subcell_size_lon = cell_box.size.lon / num_subcells
    result = []

    subcell_imgs = []

    for y in range(num_subcells):
        for x in range(num_subcells):
            subcell_img = orto_img[
                y * YOLO_MODEL_RESOLUTION:(y + 1) * YOLO_MODEL_RESOLUTION,
                x * YOLO_MODEL_RESOLUTION:(x + 1) * YOLO_MODEL_RESOLUTION,
                :
            ]
            save_image(subcell_img, f'subcell_{y}_{x}')
            subcell_imgs.append(subcell_img)

    subcell_preds = yolo_model.predict_multi(np.stack(subcell_imgs))

    for y in range(num_subcells):
        subcell_lat = cell_box.point.lat + cell_box.size.lat - subcell_size_lat * (y + 1)
        for x in range(num_subcells):
            subcell_lon = cell_box.point.lon + x * subcell_size_lon
            subcell = Box(LatLon(subcell_lat, subcell_lon), LatLon(subcell_size_lat, subcell_size_lon))
            subcell_pred = subcell_preds[y * num_subcells + x]

            boxes = subcell_pred['boxes']

            for box in boxes:
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
                        p1, p2 = transform_px_to_rad(
                            ((y_max, x_min), (y_min, x_max)), box_cell,
                            (YOLO_MODEL_RESOLUTION, YOLO_MODEL_RESOLUTION))
                        result.append(Box(p1, p2 - p1).extend(meters=CROSSING_BOX_EXTEND))
                        break

                    tx = center_x - YOLO_MODEL_RESOLUTION / 2
                    ty = center_y - YOLO_MODEL_RESOLUTION / 2

                    tx_p = tx / YOLO_MODEL_RESOLUTION
                    ty_p = ty / YOLO_MODEL_RESOLUTION

                    box_cell_translate = LatLon(
                        -ty_p * subcell.size.lat,
                        tx_p * subcell.size.lon)

                    box_cell = Box(box_cell.point + box_cell_translate, box_cell.size)

                    box_img = fetch_orto(box_cell, YOLO_MODEL_RESOLUTION)
                    box_img = normalize_yolo_image(box_img)

                    subcell_pred = yolo_model.predict_single(box_img)
                    boxes = subcell_pred['boxes']

                    if not boxes:
                        break

                    centered_i = index_box_centered(boxes, YOLO_MODEL_RESOLUTION)
                    box = boxes[centered_i]

    if result:
        print(f'[YOLO] 👁 Found {len(result)} interesting boxes')

    return result


def _filter_added_boxes(boxes: Sequence[Box]) -> Sequence[Box]:
    mask = filter_not_added(tuple(b.center() for b in boxes))
    return tuple(b for b, m in zip(boxes, mask) if m)


def _process_interesting_box(box: Box) -> CrossingSuggestion | None:
    attrib_model = _get_attrib_model()

    position = box.center()
    orto_box = Box(position, LatLon(0, 0))
    orto_box = orto_box.extend(meters=ATTRIB_POSITION_EXTEND)

    orto_img = fetch_orto(orto_box, ATTRIB_MODEL_RESOLUTION)
    if orto_img is None:
        return None

    image = normalize_attrib_image(orto_img)

    center_x = image.shape[1] / 2
    center_y = image.shape[0] / 2
    center = (center_y, center_x)

    rr, cc = draw.disk(center, radius=6, shape=image.shape[:2])
    image[rr, cc] = (0, 0, 0)
    rr, cc = draw.disk(center, radius=4, shape=image.shape[:2])
    image[rr, cc] = (1, 0, 0)

    save_image(image, 'attrib', force=True)

    image = image * 2 - 1  # MobileNet requires [-1, 1] input

    classification = attrib_model.predict_single(image)

    if not classification.is_valid:
        mark_added((position,), reason='invalid')
        return None

    return CrossingSuggestion(
        box=box,
        crossing_type=classification.crossing_type,
    )


def _submit_processed(osm: OpenStreetMap, instructions: Sequence[CrossingMergeInstructions], *, force: bool = False) -> int:
    if not (
        len(instructions) >= MIN_IMPORT_SIZE or
        (len(instructions) > 0 and force)
    ):
        return 0

    positions = tuple(i.position for i in instructions)
    tree = BallTree(positions, metric='haversine')
    query = tree.query_radius(positions, meters_to_lat(ADDED_SEARCH_RADIUS))

    added: set[int] = set()

    for i, query_indices in enumerate(query):
        # deduplicate
        if len(query_indices) > 1 and any(i in added for i in query_indices):
            continue

        added.add(i)

    if len(instructions) != len(added):
        print(f'[UPLOAD] Removed {len(instructions) - len(added)} duplicates')

    added_instructions = tuple(instructions[i] for i in added)
    added_positions = tuple(i.position for i in added_instructions)

    with print_run_time('Create OSM change'):
        osm_change = create_instructed_change(added_instructions)

    with print_run_time('Upload OSM change'):
        if DRY_RUN:
            print('[DRY-RUN] 🚫 Skipping upload')
            changeset_id = 0
        else:
            changeset_id = osm.upload_osm_change(osm_change)

    mark_added(added_positions, reason='added', changeset_id=changeset_id)
    print(f'[UPLOAD] ✅ Import successful: https://www.openstreetmap.org/changeset/{changeset_id} ({len(added)})')
    return len(added)


async def main() -> None:
    set_nice(_PROCESS_NICE)

    with print_run_time('Logging in'):
        osm = OpenStreetMap()
        display_name = osm.get_authorized_user()['display_name']
        print(f'👤 Welcome, {display_name}!')
        # changeset_max_size = osm.get_changeset_max_size()

    while True:
        processed: list[CrossingMergeInstructions] = []
        cells_gen = iter_grid()

        while (process_cells := tuple(islice(cells_gen, CPU_COUNT))):
            async def fetch_orto_with_cell(cell: Cell):
                return cell, await fetch_orto_async(cell.box, YOLO_MODEL_RESOLUTION)

            fetch_orto_tasks = tuple(
                asyncio.create_task(fetch_orto_with_cell(cell))
                for cell in process_cells)

            for task in asyncio.as_completed(fetch_orto_tasks):
                cell, orto_img = await task
                if orto_img is None:
                    print('[CELL] ⏭️ Nothing to do: missing ortophoto')
                    continue

                with print_run_time('Processing ortophoto'):
                    yolo_img = normalize_yolo_image(orto_img)
                    interesting_boxes = _process_orto(cell.box, yolo_img)

                if not interesting_boxes:
                    print(f'[CELL] ⏭️ Nothing to do: no interesting boxes')
                    continue

                with print_run_time('Filtering added boxes'):
                    interesting_boxes = _filter_added_boxes(interesting_boxes)

                if not interesting_boxes:
                    print('[CELL] ⏭️ Nothing to do: no new boxes')
                    continue

                # TODO: predict_multi
                with print_run_time('Generating suggestions'):
                    suggestions = tuple(filter(None, map(
                        lambda b: _process_interesting_box(b),
                        interesting_boxes)))

                if not suggestions:
                    print('[CELL] ⏭️ Nothing to do: no suggestions')
                    continue

                print(f'[CELL] 💡 Suggested {len(suggestions)} crossings')

                with print_run_time('Merging crossings'):
                    instructions = merge_crossings(suggestions)

                empty_mask = tuple(not i.to_nodes_ids and not i.to_ways_inst for i in instructions)
                empty_instructions = tuple(i for i, m in zip(instructions, empty_mask) if m)
                valid_instructions = tuple(i for i, m in zip(instructions, empty_mask) if not m)
                mark_added(tuple(i.position for i in empty_instructions), reason='empty')

                if valid_instructions:
                    print(f'[CELL] 📦 Processed: {len(processed)} + {len(valid_instructions)}')
                    processed.extend(valid_instructions)

                    a = DATA_DIR / 'processed.json'
                    a.write_text(json.dumps([i.position for i in processed], indent=2))

            if not processed:
                set_last_cell(process_cells[-1])
            elif (submit_size := _submit_processed(osm, processed)):
                set_last_cell(process_cells[-1])
                sleep_after_import(submit_size)
                processed.clear()

        submit_size = _submit_processed(osm, processed, force=True)
        set_last_cell(None)
        sleep_after_import(submit_size)

        if SLEEP_AFTER_GRID_ITER:
            print(f'[SLEEP-GRID] 💤 Sleeping for {SLEEP_AFTER_GRID_ITER} seconds...')
            sleep(SLEEP_AFTER_GRID_ITER)


if __name__ == '__main__':
    asyncio.run(main())
