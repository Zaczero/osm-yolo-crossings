from typing import Sequence

import numpy as np
import tensorflow as tf
from keras.models import Model

from config import ATTRIB_CONFIDENCES, ATTRIB_MODEL_PATH


class AttribTunedModel:
    def __init__(self):
        from attrib_model import get_attrib_model
        self._model: Model = get_attrib_model()
        self._model.load_weights(str(ATTRIB_MODEL_PATH))

    def predict_single(self, X: np.ndarray, min_confidences: Sequence[float] = ATTRIB_CONFIDENCES) -> tuple[Sequence[bool], Sequence[float]]:
        with tf.device('/CPU:0'):  # force CPU to better understand real performance
            pred_proba: dict = self._model.predict(X[np.newaxis, ...])[0]

        assert len(pred_proba) == len(min_confidences)
        pred = pred_proba >= np.array(min_confidences)

        return tuple(pred), tuple(pred_proba)
