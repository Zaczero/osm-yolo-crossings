from typing import NamedTuple

import cv2
import numpy as np
from skimage import (color, exposure, filters, img_as_float, img_as_ubyte,
                     morphology)
from skimage.color.adapt_rgb import adapt_rgb, each_channel

from utils import save_image


class ProcessPolygonResult(NamedTuple):
    image: np.ndarray
    mask: np.ndarray
    overlay: np.ndarray


@adapt_rgb(each_channel)
def median_rgb(image: np.ndarray, *args, **kwargs) -> np.ndarray:
    return filters.median(image, *args, **kwargs)


def normalize_yolo_image(image: np.ndarray) -> np.ndarray:
    save_image(image, 'normalize_yolo_0')
    image = median_rgb(image, morphology.disk(3))
    image = filters.unsharp_mask(image, radius=5, amount=1.5, channel_axis=2)
    image = img_as_ubyte(image[:, :, ::-1])
    image = cv2.fastNlMeansDenoisingColored(image, None, 3, 5, 5, 7)
    image = img_as_float(image[:, :, ::-1])
    save_image(image, 'normalize_yolo_1')
    return image


def normalize_attrib_image(image: np.ndarray) -> np.ndarray:
    save_image(image, 'normalize_attrib_0')
    image = normalize_yolo_image(image)

    hsv: np.ndarray = color.rgb2hsv(image)
    v_channel = hsv[:, :, 2]
    p02 = np.percentile(v_channel, 2)
    p98 = np.percentile(v_channel, 98)
    v_channel = exposure.rescale_intensity(v_channel, in_range=(p02, p98))
    hsv[:, :, 2] = v_channel
    image = color.hsv2rgb(hsv)

    save_image(image, 'normalize_attrib_1')
    return image
