from typing import NamedTuple

import cv2
import numpy as np
from skimage import (color, draw, exposure, filters, img_as_float,
                     img_as_ubyte, morphology, restoration, transform)

from box import Box
from orto import fetch_orto
from polygon import Polygon
from transform_geo_px import transform_rad_to_px
from utils import save_image

_FETCH_EXTEND = 8
_RESOLUTION = 224


class ProcessPolygonResult(NamedTuple):
    image: np.ndarray
    mask: np.ndarray
    overlay: np.ndarray


def normalize_image(image: np.ndarray) -> np.ndarray:
    save_image(image, 'normalize_0')
    image = img_as_ubyte(image[:, :, ::-1])
    image = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    image = img_as_float(image[:, :, ::-1])
    image = filters.unsharp_mask(image, radius=10, amount=1.3, channel_axis=2)
    save_image(image, 'normalize_1')
    return image


def make_polygon_mask(polygon: Polygon, image_box: Box, image_shape: tuple[int, int]) -> np.ndarray:
    antialiasing = 4
    image_shape = (image_shape[0] * antialiasing, image_shape[1] * antialiasing, image_shape[2])
    points = transform_rad_to_px(polygon, image_box, image_shape)
    mask = draw.polygon2mask(image_shape, points)[..., 0]
    mask = mask.astype(float)
    mask = transform.rescale(mask, 1 / antialiasing)
    return mask


def process_polygon(polygon: Polygon, raw_img: np.ndarray | None = None) -> ProcessPolygonResult:
    polygon_box = polygon.get_bounding_box()
    image_box = polygon_box.extend(meters=_FETCH_EXTEND).squarify()

    if raw_img is None:
        raw_img = img_as_float(fetch_orto(image_box))

    orto_img = raw_img
    assert orto_img.shape[0] == orto_img.shape[1]
    assert orto_img.dtype == float
    save_image(orto_img, '1')

    mask_img = make_polygon_mask(polygon, image_box, orto_img.shape)
    save_image(mask_img, '2')

    orto_img = normalize_image(orto_img)
    save_image(orto_img, '3')

    mask_img_binary = mask_img[:, :] > 0.5
    mask_img_contour = morphology.binary_dilation(mask_img_binary) ^ mask_img_binary

    overlay_img = np.copy(orto_img)
    overlay_img[mask_img_contour] = (1, 0, 0)
    save_image(overlay_img, '4')

    return ProcessPolygonResult(
        image=raw_img,
        mask=mask_img,
        overlay=overlay_img)


def process_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    assert len(mask.shape) == 2, 'Mask must be grayscale'
    assert image.shape[:2] == mask.shape, 'Image and mask must have the same shape'
    result = normalize_image(image)

    # calculate the distance from each point in the mask to the nearest zero
    # distances = distance_transform_edt(mask_bold < 0.5)
    # normalized_distances = 1 - (distances / max_distance)
    # normalized_distances = np.clip(normalized_distances, 0.1, 1)

    # multiply the image by the distance mask
    # result = image * normalized_distances[..., np.newaxis]

    # add mask outline
    mask_bold = morphology.dilation(mask, morphology.disk(3))
    mask_outline = mask_bold - mask
    mask_outline = np.clip(mask_outline, 0, 1)
    mask_outline = mask_outline[..., np.newaxis]
    result = result * (1 - mask_outline) + mask_outline
    # result[..., 2] = 0.5 * result[..., 2] * (1 - mask_outline) + mask_outline

    # result[..., :2] = 0
    # save_image(result, force=True)

    result = transform.resize(result, (_RESOLUTION, _RESOLUTION), anti_aliasing=True)
    result = result * 2 - 1  # MobileNet requires [-1, 1] input
    return result
