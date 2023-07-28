from math import degrees, isclose
from typing import NamedTuple, Sequence

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from config import (NODE_MERGE_THRESHOLD, ROAD_VALID_MAX_ANGLE,
                    ROAD_VALID_MAX_COUNT)
from crossing_suggestion import CrossingSuggestion
from crossing_type import CrossingType
from latlon import LatLon
from overpass import query_roads_and_crossings_historical
from utils import haversine_distance, make_way_geometry


class CrossingMergeToWayInstructions(NamedTuple):
    way_id: int
    position: LatLon
    after_node_id: int
    before_node_id: int


class CrossingMergeInstructions(NamedTuple):
    position: LatLon
    crossing_type: CrossingType
    to_nodes_ids: Sequence[int]
    to_ways_inst: Sequence[CrossingMergeToWayInstructions]


def merge_crossings(suggestions: Sequence[CrossingSuggestion]) -> Sequence[CrossingMergeInstructions]:
    roads_and_crossings = query_roads_and_crossings_historical(tuple(s.box for s in suggestions))
    result = tuple(CrossingMergeInstructions(s.box.center(), s.crossing_type, [], []) for s in suggestions)

    for i, (rac, s) in enumerate(zip(roads_and_crossings, suggestions)):
        rac_current = rac[0]
        s_box_center = s.box.center()
        skip = False

        # check for pre-existing crossings
        for rac_h in rac:
            for c in rac_h.crossings:
                if rac_h.nodes[c['id']] in s.box:
                    print(f'[MERGE] Skipping {s_box_center}: nearby crossings')
                    skip = True
                    break
            if skip:
                break
        if skip:
            continue

        # find the closest way and the point on it that's closest to the s_box_center
        closest_way = None
        closest_way_geom = None
        closest_point = None
        min_distance = float('inf')
        for way in rac_current.roads:
            way_geom = make_way_geometry(way, rac_current.nodes)
            point_on_way, _ = nearest_points(LineString(way_geom), Point(s_box_center))
            distance = haversine_distance((point_on_way.x, point_on_way.y), s_box_center)

            if distance < min_distance:
                min_distance = distance
                closest_way = way
                closest_way_geom = way_geom
                closest_point = point_on_way

        if closest_way is None:
            print(f'[MERGE] Skipping {s_box_center}: no roads')
            continue

        # find two points on the closest way that the closest point lies between
        for p1, p2 in zip(closest_way_geom, closest_way_geom[1:]):
            if isclose(LineString([p1, p2]).distance(closest_point), 0, abs_tol=1e-8):
                # calculate the unit direction vector of the closest way segment
                closest_dir_vector = np.array(p2) - np.array(p1)
                closest_dir_vector /= np.linalg.norm(closest_dir_vector)
                break

        # create a perpendicular section to the closest way segment
        perpendicular_dir_vector = np.array([-closest_dir_vector[1], closest_dir_vector[0]])
        closest_point_arr = np.array((closest_point.x, closest_point.y))
        section_length = max(s.box.size)
        section_p1 = closest_point_arr - section_length / 2 * perpendicular_dir_vector
        section_p2 = closest_point_arr + section_length / 2 * perpendicular_dir_vector
        section_line = LineString([section_p1, section_p2])

        # find all perpendicular positions
        perpendicular_positions: list[tuple[Point, str]] = [(closest_point, closest_way['id'])]
        for way in rac_current.roads:
            if way['id'] == closest_way['id']:
                continue

            way_geom = make_way_geometry(way, rac_current.nodes)
            way_line = LineString(way_geom)
            if not way_line.intersects(section_line):
                continue

            intersection = way_line.intersection(section_line)
            if intersection.geom_type == 'Point':  # single intersection point
                perpendicular_positions.append((intersection, way['id']))
            elif intersection.geom_type == 'MultiPoint':  # multiple intersection points
                perpendicular_positions.extend((point, way['id']) for point in intersection)

        # check for maximum count
        if len(perpendicular_positions) > ROAD_VALID_MAX_COUNT:
            print(f'[MERGE] Skipping {s_box_center}: too many valid roads')
            continue

        # check for maximum angle
        for intersection, way_id in perpendicular_positions:
            way = next(way for way in rac_current.roads if way['id'] == way_id)
            way_geom = make_way_geometry(way, rac_current.nodes)
            for p1, p2 in zip(way_geom, way_geom[1:]):
                if isclose(LineString([p1, p2]).distance(Point(intersection)), 0, abs_tol=1e-8):
                    way_dir_vector = np.array(p2) - np.array(p1)
                    way_dir_vector /= np.linalg.norm(way_dir_vector)

                    angle = np.arccos(np.clip(np.dot(closest_dir_vector, way_dir_vector), -1.0, 1.0))
                    angle = degrees(angle)

                    # make angle into [0, 90] range
                    if angle > 90:
                        angle = 180 - angle

                    if angle > ROAD_VALID_MAX_ANGLE:
                        print(f'[MERGE] Skipping {s_box_center}: too large angle')
                        skip = True
                        break
            if skip:
                break
        if skip:
            continue

        # create merge instructions
        for intersection, way_id in perpendicular_positions:
            way = next(way for way in rac_current.roads if way['id'] == way_id)
            way_geom = make_way_geometry(way, rac_current.nodes)
            for p1, p2 in zip(way_geom, way_geom[1:]):
                if isclose(LineString([p1, p2]).distance(Point(intersection)), 0, abs_tol=1e-8):
                    after_distance = haversine_distance((intersection.x, intersection.y), p1)
                    after_node_id = way['nodes'][way_geom.index(p1)]
                    before_distance = haversine_distance((intersection.x, intersection.y), p2)
                    before_node_id = way['nodes'][way_geom.index(p2)]
                    break

            # merge to the closest node
            if after_distance < NODE_MERGE_THRESHOLD or before_distance < NODE_MERGE_THRESHOLD:
                to_node_id = after_node_id if after_distance < before_distance else before_node_id
                result[i].to_nodes_ids.append(to_node_id)

            # merge to the way (new node)
            else:
                result[i].to_ways_inst.append(CrossingMergeToWayInstructions(
                    way_id=way_id,
                    position=LatLon(intersection.x, intersection.y),
                    after_node_id=after_node_id,
                    before_node_id=before_node_id
                ))

    return result

    # 1. for each rac (current + historical):
    # 1.1. for each way in the box, check if there is a crossing < ROAD_VALID_MIN_CROSSING_DISTANCE meters away from the s.box.center()
    # 1.2. if yes, skip this suggestion completely
    # 2. then for current:
    # 2.1. find position on the closest way, which is closest to the s.box.center() (not necessarily an existing node)
    # 2.2. find all other positions which are perpendicular (90deg) to that position, in all ways
    # 2.2.1. max perpendicular way distance should must be max(s.box.width, s.box.height), half on each side of the way
    # 2.3. if there are in total > ROAD_VALID_MAX_COUNT such positions, skip this suggestion completely
    # 2.4. if any of the matched ways is > ROAD_VALID_MAX_ANGLE angle difference from the closest way, skip this suggestion completely
    # 3. for each matched perpendicular position:
    # 3.1. attempt to merge that position to a nearby node (at the same way) if distance is < NODE_MERGE_THRESHOLD
    # 3.2. generate merge instructions for position, if merged to a node, use to_nodes_ids, otherwise use to_ways_inst
