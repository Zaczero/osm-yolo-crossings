from typing import NamedTuple, Sequence

from crossing_suggestion import CrossingSuggestion
from crossing_type import CrossingType
from latlon import LatLon
from overpass import query_roads_and_crossings_historical


class CrossingMergeToWayInstructions(NamedTuple):
    way_id: str
    position: LatLon
    after_node_id: str | None
    before_node_id: str | None


class CrossingMergeInstructions(NamedTuple):
    crossing_type: CrossingType
    to_nodes_ids: Sequence[str]
    to_ways_inst: Sequence[CrossingMergeToWayInstructions]


def merge_crossings(suggestions: Sequence[CrossingSuggestion]) -> Sequence[CrossingMergeInstructions]:
    roads_and_crossings = query_roads_and_crossings_historical(tuple(s.box for s in suggestions))
