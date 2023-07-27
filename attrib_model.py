from datetime import datetime
from math import ceil
from typing import Sequence

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV3Large, MobileNetV3Small
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard, TerminateOnNaN)
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input
from keras.losses import (BinaryCrossentropy, BinaryFocalCrossentropy,
                          CategoricalCrossentropy)
from keras.metrics import AUC, FBetaScore, Precision
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras_cv.losses import FocalLoss
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from attrib_dataset import AttribDatasetEntry, iter_attrib_dataset
from config import (ATTRIB_CONFIDENCES, ATTRIB_MODEL_PATH, ATTRIB_NUM_CLASSES,
                    DATA_DIR, SEED)
from model_save_fix import model_save_fix
from one_cycle_scheduler import OneCycleScheduler

_BATCH_SIZE = 32


def _split_x_y(dataset: Sequence[AttribDatasetEntry]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack(tuple(map(lambda x: x.image, dataset)))
    y = np.array(tuple(map(lambda x: x.labels.encode(), dataset)), dtype=float)
    return X, y


def create_attrib_model():
    dataset = tuple(iter_attrib_dataset())

    train, temp = train_test_split(dataset,
                                   test_size=0.3,
                                   random_state=SEED,
                                   stratify=tuple(map(lambda x: x.labels.encode_num(), dataset)))

    holdout, test = train_test_split(temp,
                                     test_size=2/3,
                                     random_state=SEED,
                                     stratify=tuple(map(lambda x: x.labels.encode_num(), temp)))

    X_train, y_train = _split_x_y(train)
    X_test, y_test = _split_x_y(test)
    X_holdout, y_holdout = _split_x_y(holdout)

    # train: 70%
    # test: 20%
    # val: 10%

    image_inputs = Input(dataset[0].image.shape)
    image_model = MobileNetV3Large(include_top=False,
                                   input_tensor=image_inputs,
                                   include_preprocessing=False)

    z = image_model(image_inputs)
    z = Flatten()(z)
    z = BatchNormalization()(z)
    z = Dropout(0.2)(z)
    z = Dense(256, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(64, activation='relu')(z)
    z = Dense(ATTRIB_NUM_CLASSES, activation='sigmoid')(z)

    model = Model(inputs=image_inputs, outputs=z)
    model.compile(
        optimizer=Adam(),
        loss=BinaryFocalCrossentropy(apply_class_balancing=True),
        metrics=[AUC(multi_label=True, num_labels=ATTRIB_NUM_CLASSES)],
    )

    model_save_fix(model)

    datagen = ImageDataGenerator(
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        rotation_range=180,
        shear_range=15,
        zoom_range=0.2,
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=True,
    )

    steps_per_epoch = ceil(len(train) / _BATCH_SIZE)
    cycle_epochs = 6

    callbacks = [
        OneCycleScheduler(0.00012, steps_per_epoch, cycle_epochs),

        EarlyStopping('val_auc', mode='max',
                      min_delta=0.001,
                      start_from_epoch=cycle_epochs,
                      patience=cycle_epochs * 2,
                      verbose=1),

        ModelCheckpoint(str(ATTRIB_MODEL_PATH), 'val_auc', mode='max',
                        initial_value_threshold=0.9,
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1),

        TensorBoard(str(DATA_DIR / 'tensorboard' / datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1),
        TerminateOnNaN(),
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=_BATCH_SIZE),
        epochs=1000,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
    )

    model.load_weights(str(ATTRIB_MODEL_PATH))

    y_pred_proba = model.predict(X_holdout)

    for i in range(ATTRIB_NUM_CLASSES):
        print(f'Class {i} statistics:\n')

        threshold = ATTRIB_CONFIDENCES[i]
        print(f'Threshold: {threshold}')

        y_true_class = y_holdout[:, i]
        y_pred_proba_class = y_pred_proba[:, i]
        y_pred_class = y_pred_proba[:, i] > threshold

        precision = precision_score(y_true_class, y_pred_class)
        print(f'Precision: {precision:.3f}')

        tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_class).ravel()
        print(f'True Negatives: {tn}')
        print(f'[❗] False Positives: {fp}')
        print(f'False Negatives: {fn}')
        print(f'[✅] True Positives: {tp}')
        print()

        for pred, proba, true, entry in sorted(zip(y_pred_class, y_pred_proba_class, y_true_class, holdout), key=lambda x: x[3].id.lower()):
            if pred != true and not true:
                print(f'FP: {entry.id!r} - {true} != {pred} [{proba:.3f}]')

        print('\n')
