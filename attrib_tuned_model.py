from typing import NamedTuple

import numpy as np
import tensorflow as tf

from config import ATTRIB_CONFIDENCE, ATTRIB_MODEL_PATH
from crossing_type import CrossingType


class AttribClassification(NamedTuple):
    is_valid: bool
    crossing_type: CrossingType


class AttribTunedModel:
    def __init__(self):
        from attrib_model import get_attrib_model
        self._model = get_attrib_model()
        self._model.load_weights(str(ATTRIB_MODEL_PATH))

    def predict_single(self, X: np.ndarray, min_confidence: float = ATTRIB_CONFIDENCE) -> AttribClassification:
        with tf.device('/CPU:0'):  # force CPU to better understand real performance
            pred_proba: dict = self._model.predict(X[np.newaxis, ...], verbose=0)[0]

        assert len(pred_proba) == 1

        is_valid = pred_proba[0] > min_confidence
        crossing_type = CrossingType.UNKNOWN

        # if (not is_uncontrolled and not is_traffic_signals) or (is_uncontrolled and is_traffic_signals):
        #     crossing_type = CrossingType.UNKNOWN
        # elif is_uncontrolled:
        #     crossing_type = CrossingType.UNCONTROLLED
        # elif is_traffic_signals:
        #     crossing_type = CrossingType.TRAFFIC_SIGNALS

        return AttribClassification(is_valid, crossing_type)
