from typing import Sequence

import numpy as np
import tensorflow as tf

from config import CPU_COUNT, YOLO_CONFIDENCE, YOLO_MODEL_PATH


class YoloTunedModel:
    def __init__(self):
        from yolo_model import get_yolo_model
        self._model = get_yolo_model(coco_weights=False)
        self._model.load_weights(str(YOLO_MODEL_PATH))

    def predict_single(self, X: np.ndarray, min_confidence: float = YOLO_CONFIDENCE) -> dict:
        return self.predict_multi(X[np.newaxis, ...], min_confidence)[0]

    def predict_multi(self, X: np.ndarray, min_confidence: float = YOLO_CONFIDENCE) -> Sequence[dict]:
        assert len(X.shape) == 4

        with tf.device('/CPU:0'):  # force CPU to better understand real performance
            pred_all: dict = self._model.predict(X, verbose=0 if (X.shape[0] == 1 or CPU_COUNT > 1) else 'auto')

        result = []

        for i in range(X.shape[0]):
            pred = {k: v[i] for k, v in pred_all.items()}

            num_detections = pred['num_detections']
            del pred['num_detections']

            boxes = pred['boxes'][:num_detections]
            confidences = pred['confidence'][:num_detections]
            classes = pred['classes'][:num_detections]

            # sort by confidence (descending)
            sort_indices = np.argsort(confidences)[::-1]
            boxes = boxes[sort_indices]
            confidences = confidences[sort_indices]
            classes = classes[sort_indices]

            new_boxes = []
            new_confidences = []
            new_classes = []

            for box, confidence, class_id in zip(boxes, confidences, classes):
                if confidence < min_confidence:
                    continue

                new_box = tuple(float(v) for v in box)
                new_confidence = confidence
                new_class_id = class_id

                new_boxes.append(new_box)
                new_confidences.append(new_confidence)
                new_classes.append(new_class_id)

            pred['boxes'] = new_boxes
            pred['confidence'] = new_confidences
            pred['classes'] = new_classes

            result.append(pred)

        return tuple(result)
