from typing import NamedTuple

from box import Box
from crossing_type import CrossingType


class CrossingSuggestion(NamedTuple):
    box: Box
    crossing_type: CrossingType
