from typing import Sequence

from shapely import geometry


def check_on_osm(classifieds: Sequence[ClassifiedBuilding]) -> tuple[Sequence[ClassifiedBuilding], Sequence[ClassifiedBuilding]]:
    building_polygons = query_buildings(tuple(cb.building.polygon.get_bounding_box() for cb in classifieds))

    found: list[ClassifiedBuilding] = []
    not_found: list[ClassifiedBuilding] = []

    for cb, polygons in zip(classifieds, building_polygons):
        building_shape = geometry.Polygon(cb.building.polygon.points)

        for polygon in polygons:
            polygon_shape = geometry.Polygon(polygon.points)

            if building_shape.intersects(polygon_shape):
                found.append(cb)
                break
        else:
            not_found.append(cb)

    return tuple(found), tuple(not_found)
