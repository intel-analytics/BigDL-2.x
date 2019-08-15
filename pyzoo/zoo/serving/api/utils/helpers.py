import numpy as np
import base64
import sys
import cv2
import time


def base64_encode_image(image_array):
    # base64 encode the input NumPy array
    return base64.b64encode(image_array).decode("utf-8")


def base64_decode_image(image_array, dtype):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        image_array = bytes(image_array, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    image_array = np.frombuffer(base64.decodestring(image_array), dtype=dtype)
    image_array = cv2.imdecode(image_array, -1)
    image_array = image_preprocess(image_array, "image_width", "image_height")
    # return the decoded image
    return image_array


def smallest_size_at_least(height, width, resize_min):
    smaller_dim = min(height, width)
    scale_ratio = resize_min / smaller_dim
    new_height = int(height * scale_ratio)
    new_width = int(width * scale_ratio)
    return new_height, new_width


def resize_image(image, height, width):
    return cv2.resize(image, (width, height))


def aspect_preserving_resize(image, resize_min):
    height, width = image.shape[0], image.shape[1]
    new_height, new_width = smallest_size_at_least(height, width, resize_min)
    return resize_image(image, new_height, new_width)


def central_crop(image, crop_height, crop_width):
    height, width = image.shape[0], image.shape[1]
    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return image[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width]


def image_preprocess(image, output_width, output_height):
    start_time = time.time()
    image = aspect_preserving_resize(image, "resize_min")
    image = central_crop(image, output_height, output_width)
    print("Pre-processing %d ms" % int(round((time.time() - start_time) * 1000)))
    # NHWC -> NCWH
    image = image.transpose(2, 0, 1)
    # ensure our NumPy array is C-contiguous as well,
    # otherwise we won't be able to serialize it
    image = image.copy(order="C")
    return image
