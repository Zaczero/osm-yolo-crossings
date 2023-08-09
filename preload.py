from attrib_model import get_attrib_model
from utils import print_run_time
from yolo_model import get_yolo_model


def preload() -> None:
    with print_run_time('Preloading models'):
        get_yolo_model(coco_weights=False)
        get_yolo_model(coco_weights=True)
        get_attrib_model(imagenet_weights=True)


if __name__ == '__main__':
    preload()
