from datetime import datetime
from math import ceil
from typing import Sequence

import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV3Large
from keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN
from keras.experimental import CosineDecay
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input
from keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy
from keras.metrics import (AUC, F1Score, FBetaScore, PrecisionAtRecall,
                           RecallAtPrecision)
from keras.models import Model
from keras.optimizers import AdamW
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from attrib_dataset import AttribDatasetEntry, iter_attrib_dataset
from config import (ATTRIB_MODEL_PATH, ATTRIB_MODEL_RESOLUTION,
                    ATTRIB_PRECISION, DATA_DIR, SEED)

_BATCH_SIZE = 32
_EPOCHS = 40


def _split_x_y(dataset: Sequence[AttribDatasetEntry]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack(tuple(map(lambda x: x.image, dataset)))
    y = np.array(tuple(map(lambda x: x.labels.is_valid, dataset)), dtype=float)
    # TODO: at some point, existence-dependant features should be moved to a 2nd stage model
    return X, y


def get_attrib_model(imagenet_weights: bool = True) -> Model:
    image_inputs = Input(shape=(ATTRIB_MODEL_RESOLUTION, ATTRIB_MODEL_RESOLUTION, 3))
    image_model = MobileNetV3Large(include_top=False,
                                   weights='imagenet' if imagenet_weights else None,
                                   input_tensor=image_inputs,
                                   dropout_rate=0.3,
                                   include_preprocessing=False)

    freeze_ratio = 0.65
    for layer in image_model.layers[:int(len(image_model.layers) * freeze_ratio)]:
        layer.trainable = False

    z = image_model(image_inputs)
    z = Flatten()(z)
    z = Dropout(0.3)(z)
    z = BatchNormalization()(z)
    z = Dense(256, activation='relu')(z)
    z = Dropout(0.3)(z)
    z = BatchNormalization()(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.3)(z)
    z = BatchNormalization()(z)
    z = Dense(64, activation='relu')(z)
    z = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=image_inputs, outputs=z)

    return model


def create_attrib_model():
    dataset = tuple(iter_attrib_dataset())

    labels = tuple(map(lambda x: x.labels.is_valid, dataset))
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weights))

    train, holdout = train_test_split(dataset,
                                      test_size=0.3,
                                      random_state=SEED,
                                      stratify=tuple(map(lambda x: x.labels.is_valid, dataset)))

    _, test = train_test_split(holdout,
                               test_size=2/3,
                               random_state=SEED,
                               stratify=tuple(map(lambda x: x.labels.is_valid, holdout)))

    X_train, y_train = _split_x_y(train)
    X_test, y_test = _split_x_y(test)
    X_holdout, y_holdout = _split_x_y(holdout)

    # train: 70%
    # test: 20%
    # val: 10%

    steps_per_epoch = ceil(len(train) / _BATCH_SIZE)

    datagen = ImageDataGenerator(
        rotation_range=180,
        shear_range=20,
        zoom_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        channel_shift_range=0.1,
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=True,
    )

    model = get_attrib_model()
    model.compile(
        optimizer=AdamW(
            CosineDecay(initial_learning_rate=2e-5,
                        decay_steps=steps_per_epoch * _EPOCHS - 5,
                        alpha=0.3,
                        warmup_target=2e-4,
                        warmup_steps=steps_per_epoch * 5)),
        loss=BinaryCrossentropy(),
        metrics=[
            AUC(),
            RecallAtPrecision(0.995),
            # PrecisionAtRecall(0.8),
            # F1Score('micro', threshold=0.5),
            # FBetaScore('micro', beta=0.5, threshold=0.5),
        ],
    )

    callbacks = [
        ModelCheckpoint(str(ATTRIB_MODEL_PATH), 'val_auc', mode='max',
                        initial_value_threshold=0.95,
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1),

        TensorBoard(str(DATA_DIR / 'tensorboard' / datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1),
        TerminateOnNaN(),
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=_BATCH_SIZE),
        epochs=_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weights,
    )

    model.load_weights(str(ATTRIB_MODEL_PATH))

    y_pred_proba = model.predict(X_holdout).flatten()
    precisions, _, thresholds = precision_recall_curve(y_holdout, y_pred_proba)
    threshold_optimal = thresholds[np.searchsorted(precisions, ATTRIB_PRECISION) - 1]
    print(f'Threshold: {threshold_optimal}')

    y_pred = y_pred_proba >= threshold_optimal

    val_score = precision_score(y_holdout, y_pred)
    print(f'Validation score: {val_score:.3f}')
    print()

    tn, fp, fn, tp = confusion_matrix(y_holdout, y_pred).ravel()
    print(f'True Negatives: {tn}')
    print(f'[❗] False Positives: {fp}')
    print(f'False Negatives: {fn}')
    print(f'[✅] True Positives: {tp}')
    print()
    print(f'Recall: {recall_score(y_holdout, y_pred):.3f}')
    print()

    for pred, proba, true, entry in sorted(zip(y_pred, y_pred_proba, y_holdout, holdout), key=lambda x: x[3].id.lower()):
        if pred != true and not true:
            print(f'FP: {entry.id!r} - {true} != {pred} [{proba:.3f}]')
