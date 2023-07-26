import tensorflow as tf
from keras.losses import BinaryCrossentropy
from keras_cv.layers import MultiClassNonMaxSuppression


class PredictionDecoderFix(MultiClassNonMaxSuppression):
    def call(self, box_prediction, class_prediction, images=None, image_shape=None):
        result = super().call(box_prediction, class_prediction, images, image_shape)
        x = 1
