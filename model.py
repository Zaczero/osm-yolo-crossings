import pickle
import random
from datetime import datetime
from functools import partial
from itertools import chain
from math import ceil
from typing import Generator, Sequence

import cv2
import keras.backend as K
import keras_cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV3Large
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input
from keras.losses import BinaryCrossentropy
from keras.metrics import Precision
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage import draw, transform
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from Xlib import X, display

from config import (CACHE_DIR, CONFIDENCE, DATA_DIR, MODEL_PATH,
                    MODEL_RESOLUTION, SEED)
from dataset import DatasetEntry, iter_dataset
from keras_box import KerasBoundingBox
from model_save_fix import model_save_fix
from processor import normalize_image
from utils import draw_predictions, save_image

_BATCH_SIZE = 32
_STEPS_PER_EPOCH = 10
_BOXES_COUNT = 4


def _setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # tf.config.set_logical_device_configuration(gpus[0], [
        #     tf.config.LogicalDeviceConfiguration(memory_limit=9 * 1024),
        # ])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')


def _data_gen(dataset: Sequence[DatasetEntry], batch_size: int = _BATCH_SIZE, *, transform: bool = True) -> Generator[tuple[np.ndarray, dict], None, None]:
    if transform:
        datagen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            rotation_range=180,
            shear_range=15,
            zoom_range=0.3,
            channel_shift_range=0.25,
            fill_mode='constant',
            cval=0,
            horizontal_flip=True,
            vertical_flip=True,
        )

    else:
        datagen = ImageDataGenerator()

    if transform and len(dataset) < batch_size:
        X_batch, y_batch = next(_data_gen(dataset, batch_size=len(dataset), transform=False))
        X_batch = X_batch.tolist()
        y_batch['boxes'] = y_batch['boxes'].tolist()
        y_batch['classes'] = y_batch['classes'].tolist()
    else:
        X_batch = []
        y_batch = {
            'boxes': [],
            'classes': [],
        }

    while True:
        seed_ = random.randint(0, 2**31 - 1)

        for i, entry in enumerate(dataset):
            seed = seed_ + i

            image = entry.image
            boxes = []
            classes = []

            params = datagen.get_random_transform(image.shape, seed=seed)

            # params['zx'] = min(1, params['zx'])
            # params['zy'] = min(1, params['zy'])

            # # prevent out of bounds
            # max_tx = image.shape[1] * (1 - params['zx']) / 2
            # max_ty = image.shape[0] * (1 - params['zy']) / 2

            # if abs(params['tx']) > max_tx:
            #     params['tx'] = np.sign(params['tx']) * max_tx

            # if abs(params['ty']) > max_ty:
            #     params['ty'] = np.sign(params['ty']) * max_ty

            for polygon, label in zip(entry.labels.polygons, entry.labels.labels):
                box = polygon.transform_and_bb(params, image.shape)
                x, y, w, h = box

                if w * h < 150:
                    continue

                boxes.append(box)
                classes.append(label)

            if not len(boxes) or _BOXES_COUNT < len(boxes):
                continue

            for _ in range(_BOXES_COUNT - len(boxes)):
                boxes.append((-1, -1, -1, -1))
                classes.append(0)

            image = datagen.apply_transform(image, params)

            # temp = image.copy()
            # for box in boxes:
            #     x, y, w, h = box
            #     rr, cc = draw.rectangle_perimeter((y, x), extent=(h, w), shape=temp.shape)
            #     temp[rr, cc] = (1, 0, 0)
            # save_image(temp, f'datagen_{i}')

            X_batch.append(image)
            y_batch['boxes'].append(boxes)
            y_batch['classes'].append(classes)

            if len(X_batch) == batch_size:
                X_batch = np.stack(X_batch)
                y_batch['boxes'] = np.array(y_batch['boxes'], float)
                y_batch['classes'] = np.array(y_batch['classes'], int)
                yield X_batch, y_batch
                X_batch = []
                y_batch['boxes'] = []
                y_batch['classes'] = []


def create_model():
    _setup_gpu()

    # dataset_iterator = iter_dataset()
    # dataset = tuple(next(dataset_iterator) for _ in range(100))
    dataset = tuple(iter_dataset())

    train, temp = train_test_split(dataset,
                                   test_size=0.3,
                                   random_state=SEED,
                                   shuffle=True)

    holdout, test = train_test_split(temp,
                                     test_size=2/3,
                                     random_state=SEED,
                                     shuffle=True)

    X_test, y_test = next(_data_gen(test, batch_size=len(test), transform=False))
    X_holdout, y_holdout = next(_data_gen(holdout, batch_size=len(holdout), transform=False))

    print(f'Train size: {len(train)}')
    print(f'Test size: {len(test)}')
    print(f'Holdout size: {len(holdout)}')

    # take samples from train to test to avoid no prediction issues which affect loss
    test = tuple(chain(train[:5], test))

    model = keras_cv.models.YOLOV8Detector(
        backbone=keras_cv.models.YOLOV8Backbone.from_preset('yolo_v8_s_backbone_coco'),
        num_classes=1,
        bounding_box_format='xywh',
    )

    model.compile(
        box_loss='ciou',
        classification_loss='binary_crossentropy',
        optimizer=tf.optimizers.Adam(
            learning_rate=0.003,
            global_clipnorm=2.0,
        )
        # optimizer=tf.optimizers.SGD(
        #     learning_rate=0.1,
        #     momentum=0.9,
        #     nesterov=True,
        #     clipnorm=2.0
        # )
    )

    model_save_fix(model)

    callbacks_early = [
        EarlyStopping('loss', patience=5, min_delta=0.3, verbose=1),
        TensorBoard(str(DATA_DIR / 'tensorboard' / datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1),
    ]

    callbacks_late = [
        EarlyStopping(patience=20, min_delta=0.01, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=10, min_delta=0.01, verbose=1),
        ModelCheckpoint(str(MODEL_PATH),
                        save_best_only=True, save_weights_only=True,
                        verbose=1, initial_value_threshold=2.5),
        callbacks_early[1],  # TensorBoard
    ]

    X_batch, y_batch = next(_data_gen(train, batch_size=_BATCH_SIZE * _STEPS_PER_EPOCH * 10))
    memory_usage = X_batch.nbytes + y_batch['boxes'].nbytes + y_batch['classes'].nbytes
    print(f'Batch memory usage: {memory_usage / 1024 / 1024:.2f} MiB')

    model.fit(
        X_batch, y_batch,
        batch_size=_BATCH_SIZE,
        steps_per_epoch=_STEPS_PER_EPOCH,
        epochs=1000,
        shuffle=False,
        callbacks=callbacks_early,
    )

    model.fit(
        X_batch, y_batch,
        batch_size=_BATCH_SIZE,
        steps_per_epoch=_STEPS_PER_EPOCH,
        epochs=1000,
        shuffle=False,
        validation_data=(X_test, y_test),
        callbacks=callbacks_late,
    )

    exit()

    model = keras_cv.models.YOLOV8Detector(
        backbone=keras_cv.models.YOLOV8Backbone.from_preset('yolo_v8_s_backbone_coco'),
        num_classes=1,
        bounding_box_format='xywh',
    )

    model.load_weights(str(MODEL_PATH))

    capture = display.Display().screen().root

    with tf.device('/CPU:0'):
        while True:
            I = capture.get_image(720, 240, 800, 800, X.ZPixmap, 0xffffffff)
            img = Image.frombytes('RGB', (800, 800), I.data, 'raw', 'BGRX')
            screenshot = np.asarray(img)
            screenshot = transform.resize(screenshot, (MODEL_RESOLUTION, MODEL_RESOLUTION, 3))
            screenshot = normalize_image(screenshot)
            pred = model.predict(screenshot[np.newaxis, ...])
            frame = draw_predictions(screenshot, pred, 0)

            save_image(frame, 'frame', force=True)

    # threshold = CONFIDENCE
    # print(f'Threshold: {threshold}')

    # y_pred = model.predict(X_holdout)
    # y_pred = y_pred_proba >= threshold

    val_score = precision_score(y_holdout, y_pred)
    print(f'Validation score: {val_score:.3f}')
    print()

    tn, fp, fn, tp = confusion_matrix(y_holdout, y_pred).ravel()
    print(f'True Negatives: {tn}')
    print(f'[❗] False Positives: {fp}')
    print(f'False Negatives: {fn}')
    print(f'[✅] True Positives: {tp}')
    print()

    for pred, proba, true, entry in sorted(zip(y_pred, y_pred_proba, y_holdout, holdout), key=lambda x: x[3].id.lower()):
        if pred != true and not true:
            print(f'FP: {entry.id!r} - {true} != {pred} [{proba:.3f}]')
