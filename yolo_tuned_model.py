import numpy as np
import tensorflow as tf

from config import YOLO_CONFIDENCE, YOLO_MODEL_PATH


class YoloTunedModel:
    def __init__(self):
        from yolo_model import get_yolo_model
        self._model = get_yolo_model()
        self._model.load_weights(str(YOLO_MODEL_PATH))

    def predict_single(self, X: np.ndarray, min_confidence: float = YOLO_CONFIDENCE) -> dict:
        with tf.device('/CPU:0'):  # force CPU to better understand real performance
            pred: dict = self._model.predict(X[np.newaxis, ...])

        for k, v in pred.items():
            pred[k] = v[0]

        num_detections = pred['num_detections']

        boxes = pred['boxes'][:num_detections]
        confidences = pred['confidence'][:num_detections]
        classes = pred['classes'][:num_detections]

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
        del pred['num_detections']

        return pred
