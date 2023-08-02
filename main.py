import asyncio
import random
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from heapq import heappop, heappush
from itertools import islice
from math import inf
from time import sleep
from typing import Sequence

import numpy as np
from skimage import draw

from attrib_dataset import create_attrib_dataset
from attrib_model import create_attrib_model
from attrib_tuned_model import AttribTunedModel
from box import Box
from config import (ATTRIB_MODEL_RESOLUTION, ATTRIB_POSITION_EXTEND, CPU_COUNT,
                    CROSSING_BOX_EXTEND, DRY_RUN, MIN_IMPORT_SIZE,
                    PROCESS_NICE, SEED, SLEEP_AFTER_GRID_ITER,
                    YOLO_MODEL_RESOLUTION)
from crossing_merger import CrossingMergeInstructions, merge_crossings
from crossing_suggestion import CrossingSuggestion
from db_added import filter_not_added, mark_added
from db_grid import Cell, iter_grid, set_last_cell_index
from import_speed_limit import ImportSpeedLimit
from latlon import LatLon
from openstreetmap import OpenStreetMap
from orto import fetch_orto, fetch_orto_async
from osm_change import create_instructed_change
from processor import normalize_attrib_image, normalize_yolo_image
from transform_geo_px import transform_px_to_rad
from utils import (index_box_centered, print_run_time, save_image, set_nice,
                   setup_gpu)
from yolo_dataset import create_yolo_dataset
from yolo_model import create_yolo_model
from yolo_tuned_model import YoloTunedModel

random.seed(SEED)
np.random.seed(SEED)

_MIN_EDGE_DISTANCE = 0.1


@cache
def _get_yolo_model() -> YoloTunedModel:
    with print_run_time('Loading YOLO model'):
        return YoloTunedModel()


@cache
def _get_attrib_model() -> AttribTunedModel:
    with print_run_time('Loading ATTRIB model'):
        return AttribTunedModel()


def _process_object_detection(cell_box: Box, orto_img: np.ndarray) -> Sequence[Box]:
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


def _process_cell(cell: Cell) -> Sequence[CrossingMergeInstructions]:
    # with print_run_time('Fetching ortophoto'):
    orto_img = fetch_orto(cell.box, YOLO_MODEL_RESOLUTION)

    if orto_img is None:
        print('[CELL] ⏭️ Nothing to do: missing ortophoto')
        return ()

    # with print_run_time('Processing ortophoto'):
    yolo_img = normalize_yolo_image(orto_img)
    interesting_boxes = _process_object_detection(cell.box, yolo_img)

    if not interesting_boxes:
        # this message prints too often
        # print(f'[CELL] ⏭️ Nothing to do: no interesting boxes')
        return ()

    with print_run_time('Filtering added boxes'):
        interesting_boxes = _filter_added_boxes(interesting_boxes)

    if not interesting_boxes:
        print('[CELL] ⏭️ Nothing to do: no new boxes')
        return ()

    # TODO: predict_multi, very low priority
    with print_run_time('Generating suggestions'):
        suggestions = tuple(filter(None, map(
            lambda b: _process_interesting_box(b),
            interesting_boxes)))

    if not suggestions:
        print('[CELL] ⏭️ Nothing to do: no suggestions')
        return ()

    print(f'[CELL] 💡 Suggested {len(suggestions)} crossings')

    with print_run_time('Merging crossings'):
        instructions = merge_crossings(suggestions)

    empty_mask = tuple(not i.to_nodes_ids and not i.to_ways_inst for i in instructions)
    empty_instructions = tuple(i for i, m in zip(instructions, empty_mask) if m)
    valid_instructions = tuple(i for i, m in zip(instructions, empty_mask) if not m)
    mark_added(tuple(i.position for i in empty_instructions), reason='empty')
    return valid_instructions


def _submit_processed(osm: OpenStreetMap, instructions: Sequence[CrossingMergeInstructions], *, force: bool = False) -> int:
    if not (
        len(instructions) >= MIN_IMPORT_SIZE or
        (len(instructions) > 0 and force)
    ):
        return 0

    with print_run_time('Create OSM change'):
        osm_change, added_positions = create_instructed_change(instructions)

    with print_run_time('Upload OSM change'):
        if DRY_RUN:
            print('[DRY-RUN] 🚫 Skipping upload')
            changeset_id = 0
        else:
            changeset_id = osm.upload_osm_change(osm_change)

    mark_added(added_positions, reason='added', changeset_id=changeset_id)
    print(f'[UPLOAD] ✅ Import successful: https://www.openstreetmap.org/changeset/{changeset_id} '
          f'({len(added_positions)})')
    return len(added_positions)


async def main() -> None:
    set_nice(PROCESS_NICE)
    import_speed_limit = ImportSpeedLimit()

    with print_run_time('Logging in'):
        osm = OpenStreetMap()
        display_name = osm.get_authorized_user()['display_name']
        print(f'👤 Welcome, {display_name}!')
        # changeset_max_size = osm.get_changeset_max_size()

    with ProcessPoolExecutor(CPU_COUNT) as executor:
        while True:
            processed: list[CrossingMergeInstructions] = []

            cells_gen = iter_grid()
            process_futures: dict[int, asyncio.Future] = {}
            processed_heap = []

            while True:
                # get process results
                for key, future in tuple(process_futures.items()):
                    if future.done():
                        process_futures.pop(key)
                        result = future.result()
                        heappush(processed_heap, (key, result))

                # submit processes
                for cell_ in islice(cells_gen, CPU_COUNT - len(process_futures)):
                    process_futures[cell_.index] = executor.submit(_process_cell, cell_)

                # process results from the queue
                while processed_heap and processed_heap[0][0] < min(process_futures, default=inf):
                    cell_index, instructions = heappop(processed_heap)

                    if instructions:
                        print(f'[CELL] 📦 Processed: {len(processed)} + {len(instructions)}')
                        processed.extend(instructions)

                        for inst in instructions:
                            print(f'[CELL] 🦓 {inst.position}: {inst.crossing_type}')

                    if not processed:
                        set_last_cell_index(cell_index)
                    elif (submit_size := _submit_processed(osm, processed)):
                        set_last_cell_index(cell_index)
                        import_speed_limit.sleep(submit_size)
                        processed.clear()

                # check if we are done
                if not process_futures:
                    assert not processed_heap
                    break

            submit_size = _submit_processed(osm, processed, force=True)
            set_last_cell_index(None)
            import_speed_limit.sleep(submit_size)

            if SLEEP_AFTER_GRID_ITER:
                print(f'[SLEEP-GRID] 💤 Sleeping for {SLEEP_AFTER_GRID_ITER} seconds...')
                sleep(SLEEP_AFTER_GRID_ITER)


if __name__ == '__main__':
    setup_gpu()
    asyncio.run(main())
