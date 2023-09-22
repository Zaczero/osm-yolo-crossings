from itertools import chain
from math import degrees, isclose
from typing import NamedTuple, Sequence

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from config import (BOX_VALID_MAX_CENTER_DISTANCE, BOX_VALID_MAX_ROAD_ANGLE,
                    BOX_VALID_MAX_ROAD_COUNT, BOX_VALID_MIN_CROSSING_DISTANCE,
                    BOX_VALID_MIN_CROSSING_DISTANCE_CONE,
                    BOX_VALID_MIN_CROSSING_DISTANCE_CONE_ANGLE,
                    MIN_SLEEP_AFTER_IMPORT, NODE_MERGE_THRESHOLD,
                    NODE_MERGE_THRESHOLD_PRIORITY)
from crossing_suggestion import CrossingSuggestion
from crossing_type import CrossingType
from latlon import LatLon
from overpass import query_roads_and_crossings_historical
from utils import (haversine_distance, lat_to_meters, lon_to_meters,
                   make_way_geometry, meters_to_lat, meters_to_lon)


class CrossingMergeToWayInstructions(NamedTuple):
    way_id: int
    position: LatLon


class CrossingMergeInstructions(NamedTuple):
    position: LatLon
    crossing_type: CrossingType
    to_nodes_ids: Sequence[int]
    to_ways_inst: Sequence[CrossingMergeToWayInstructions]


class PerpendicularPosition(NamedTuple):
    point: Point
    way_id: int
    way: dict
    way_geom: Sequence[LatLon]
    p1: LatLon
    p2: LatLon
    way_direction_vector: np.ndarray

    @property
    def position(self) -> LatLon:
        return LatLon(self.point.x, self.point.y)


def merge_crossings(suggestions: Sequence[CrossingSuggestion]) -> Sequence[CrossingMergeInstructions]:
    queried = query_roads_and_crossings_historical(tuple(s.box for s in suggestions), max_age=MIN_SLEEP_AFTER_IMPORT)
    result = tuple(CrossingMergeInstructions(s.box.center(), s.crossing_type, [], []) for s in suggestions)

    for i, (rac, s) in enumerate(zip(queried, suggestions)):
        rac_current = rac[0]
        paths_nodes_ids = set(chain.from_iterable(p['nodes'] for p in rac_current.paths))
        s_box_center = s.box.center()
        skip = False

        # find the closest way and the point on it that's closest to the s_box_center
        closest_way = None
        closest_way_geom = None
        closest_point = None
        min_distance = BOX_VALID_MAX_CENTER_DISTANCE
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
                delta_y = lat_to_meters(p2[0] - p1[0])
                delta_x = lon_to_meters(p2[1] - p1[1], (p2[0] + p1[0]) / 2)
                closest_dir_vector = np.array([delta_y, delta_x])
                closest_dir_vector /= np.linalg.norm(closest_dir_vector)
                break

        # create a perpendicular direction to the closest way segment
        perpendicular_dir_vector = (-closest_dir_vector[1], closest_dir_vector[0])

        # convert the perpendicular direction back to latitude-longitude scale
        lat_adjustment = meters_to_lat(BOX_VALID_MAX_CENTER_DISTANCE) * perpendicular_dir_vector[0]
        lon_adjustment = meters_to_lon(BOX_VALID_MAX_CENTER_DISTANCE, s_box_center[0]) * perpendicular_dir_vector[1]

        section_p1 = (s_box_center[0] - lat_adjustment, s_box_center[1] - lon_adjustment)
        section_p2 = (s_box_center[0] + lat_adjustment, s_box_center[1] + lon_adjustment)
        section_line = LineString([section_p1, section_p2])

        # find all perpendicular positions
        perpendicular_positions: list[PerpendicularPosition] = []
        for way in rac_current.roads:
            way_geom = make_way_geometry(way, rac_current.nodes)
            way_line = LineString(way_geom)
            if not way_line.intersects(section_line):
                continue

            def make_position(p: Point) -> PerpendicularPosition:
                for p1, p2 in zip(way_geom, way_geom[1:]):
                    if isclose(LineString([p1, p2]).distance(p), 0, abs_tol=1e-8):
                        delta_y = lat_to_meters(p2[0] - p1[0])
                        delta_x = lon_to_meters(p2[1] - p1[1], (p2[0] + p1[0]) / 2)
                        way_direction_vector = np.array([delta_y, delta_x])
                        way_direction_vector /= np.linalg.norm(way_direction_vector)
                        return PerpendicularPosition(p, way['id'], way, way_geom, p1, p2, way_direction_vector)

            intersection = way_line.intersection(section_line)
            if intersection.geom_type == 'Point':  # single intersection point
                perpendicular_positions.append(make_position(intersection))
            elif intersection.geom_type == 'MultiPoint':  # multiple intersection points
                perpendicular_positions.extend(make_position(point) for point in intersection.geoms)

        # check for maximum count
        if len(perpendicular_positions) > BOX_VALID_MAX_ROAD_COUNT:
            print(f'[MERGE] Skipping {s_box_center}: too many valid roads')
            continue

        # check for maximum angle
        for pp in perpendicular_positions:
            angle = np.arccos(np.clip(np.dot(closest_dir_vector, pp.way_direction_vector), -1.0, 1.0))
            angle = degrees(angle)

            # make angle into [0, 90] range
            if angle > 90:
                angle = 180 - angle

            if angle > BOX_VALID_MAX_ROAD_ANGLE:
                print(f'[MERGE] Skipping {s_box_center}: too large angle')
                skip = True
                break
        if skip:
            continue

        # check for nearby crossings
        def has_nearby_crossing(pp: PerpendicularPosition) -> bool:
            for rac_h in rac:
                for crossing_node in rac_h.crossings:
                    crossing_position = rac_h.nodes[crossing_node['id']]
                    crossing_distance = haversine_distance(crossing_position, pp.position)
                    if crossing_distance < BOX_VALID_MIN_CROSSING_DISTANCE:
                        return True

                    if crossing_distance < BOX_VALID_MIN_CROSSING_DISTANCE_CONE:
                        # calculate the angle between the crossing vector and the road direction
                        delta_y = lat_to_meters(crossing_position[0] - pp.position[0])
                        delta_x = lon_to_meters(crossing_position[1] - pp.position[1],
                                                (crossing_position[0] + pp.position[0]) / 2)

                        crossing_vector = np.array([delta_y, delta_x])
                        crossing_vector /= np.linalg.norm(crossing_vector)

                        crossing_angle = np.arccos(np.clip(np.dot(pp.way_direction_vector, crossing_vector), -1.0, 1.0))
                        crossing_angle = degrees(crossing_angle)

                        # make angle into [0, 90] range
                        if crossing_angle > 90:
                            crossing_angle = 180 - crossing_angle

                        # check if the crossing lies within the forward or backward cone
                        if crossing_angle < BOX_VALID_MIN_CROSSING_DISTANCE_CONE_ANGLE:
                            return True
            return False

        perpendicular_positions = list(filter(lambda pp: not has_nearby_crossing(pp), perpendicular_positions))

        if not perpendicular_positions:
            print(f'[MERGE] Skipping {s_box_center}: nearby crossings')
            continue

        # create merge instructions
        for pp in perpendicular_positions:
            after_node_id = pp.way['nodes'][pp.way_geom.index(pp.p1)]
            after_distance = haversine_distance(pp.position, pp.p1)
            after_priority = after_node_id in paths_nodes_ids

            before_node_id = pp.way['nodes'][pp.way_geom.index(pp.p2)]
            before_distance = haversine_distance(pp.position, pp.p2)
            before_priority = before_node_id in paths_nodes_ids

            if after_priority and before_priority and (
                    after_distance < NODE_MERGE_THRESHOLD_PRIORITY or before_distance < NODE_MERGE_THRESHOLD_PRIORITY):
                closest_node_id = after_node_id if after_distance < before_distance else before_node_id
                result[i].to_nodes_ids.append(closest_node_id)

            elif after_priority and after_distance < NODE_MERGE_THRESHOLD_PRIORITY:
                result[i].to_nodes_ids.append(after_node_id)

            elif before_priority and before_distance < NODE_MERGE_THRESHOLD_PRIORITY:
                result[i].to_nodes_ids.append(before_node_id)

            # merge to the closest node
            elif after_distance < NODE_MERGE_THRESHOLD or before_distance < NODE_MERGE_THRESHOLD:
                closest_node_id = after_node_id if after_distance < before_distance else before_node_id
                result[i].to_nodes_ids.append(closest_node_id)

            # merge to the way (new node)
            else:
                result[i].to_ways_inst.append(CrossingMergeToWayInstructions(
                    way_id=pp.way_id,
                    position=pp.position))

    return result
