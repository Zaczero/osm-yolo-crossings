import random
from datetime import datetime
from math import ceil
from typing import Generator, Sequence

import keras_cv
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN
from keras.experimental import CosineDecay
from keras.losses import BinaryFocalCrossentropy
from keras.models import Model
from keras.optimizers import AdamW
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage import transform
from sklearn.model_selection import train_test_split
from Xlib import X, display

from config import DATA_DIR, SEED, YOLO_MODEL_PATH, YOLO_MODEL_RESOLUTION
from model_save_fix import model_save_fix
from processor import normalize_yolo_image
from utils import draw_predictions, save_image
from yolo_dataset import YoloDatasetEntry, iter_yolo_dataset

_EPOCHS_0 = 100
_EPOCHS = 1000
_BATCH_SIZE = 32
_BOXES_COUNT = 4


def _data_gen(dataset: Sequence[YoloDatasetEntry], batch_size: int = _BATCH_SIZE, *, transform: bool = True) -> Generator[tuple[np.ndarray, dict], None, None]:
    if transform:
        datagen = ImageDataGenerator(
            rotation_range=180,
            shear_range=10,
            zoom_range=0.1,
            width_shift_range=0.15,
            height_shift_range=0.15,
            channel_shift_range=0.1,
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


def get_yolo_model(coco_weights: bool = True) -> Model:
    preset = 'yolo_v8_xs_backbone'

    if coco_weights:
        preset += '_coco'

    model = keras_cv.models.YOLOV8Detector(
        backbone=keras_cv.models.YOLOV8Backbone.from_preset(preset),
        bounding_box_format='xywh',
        num_classes=1,
    )

    return model


def create_yolo_model():
    # dataset_iterator = iter_yolo_dataset()
    # dataset = tuple(next(dataset_iterator) for _ in range(100))
    dataset = tuple(iter_yolo_dataset())

    train, test = train_test_split(dataset,
                                   test_size=0.3,
                                   random_state=SEED,
                                   shuffle=True)

    # train: 70%
    # test: 30%

    X_test, y_test = next(_data_gen(test, batch_size=len(test), transform=False))

    print(f'Train size: {len(train)}')
    print(f'Test size: {len(test)}')

    steps_per_epoch = ceil(len(train) / _BATCH_SIZE)

    model = get_yolo_model()

    model_save_fix(model)
    model.compile(
        optimizer=AdamW(
            CosineDecay(initial_learning_rate=5e-4,
                        decay_steps=steps_per_epoch * (_EPOCHS - _EPOCHS_0),
                        alpha=0.01,
                        warmup_target=5e-5,
                        warmup_steps=steps_per_epoch * _EPOCHS_0)),
        box_loss='ciou',
        classification_loss=BinaryFocalCrossentropy(apply_class_balancing=True),
    )

    callbacks = [
        ModelCheckpoint(str(YOLO_MODEL_PATH),
                        initial_value_threshold=1.5,
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1),

        TensorBoard(str(DATA_DIR / 'tensorboard' / datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1),
        TerminateOnNaN(),
    ]

    X_batch, y_batch = next(_data_gen(train, batch_size=_BATCH_SIZE * steps_per_epoch * 10))
    memory_usage = X_batch.nbytes + y_batch['boxes'].nbytes + y_batch['classes'].nbytes
    print(f'Batch memory usage: {memory_usage / 1024 / 1024:.2f} MiB')

    model.fit(
        X_batch, y_batch,
        batch_size=_BATCH_SIZE,
        steps_per_epoch=steps_per_epoch,
        epochs=_EPOCHS_0,
        shuffle=False,
    )

    model.fit(
        X_batch, y_batch,
        batch_size=_BATCH_SIZE,
        steps_per_epoch=steps_per_epoch,
        epochs=_EPOCHS - _EPOCHS_0,
        shuffle=False,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
    )

    exit()

    model.load_weights(str(YOLO_MODEL_PATH))

    capture = display.Display().screen().root

    with tf.device('/CPU:0'):
        while True:
            I = capture.get_image(720, 240, 800, 800, X.ZPixmap, 0xffffffff)
            img = Image.frombytes('RGB', (800, 800), I.data, 'raw', 'BGRX')
            screenshot = np.asarray(img)
            screenshot = transform.resize(screenshot, (YOLO_MODEL_RESOLUTION, YOLO_MODEL_RESOLUTION, 3))
            screenshot = normalize_yolo_image(screenshot)
            pred = model.predict(screenshot[np.newaxis, ...])
            frame = draw_predictions(screenshot, pred, 0)

            save_image(frame, 'frame', force=True)
