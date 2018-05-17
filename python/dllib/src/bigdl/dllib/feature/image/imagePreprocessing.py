#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from bigdl.util.common import *
from zoo.feature.common import Preprocessing
from zoo.feature.image.imageset import ImageSet

if sys.version >= '3':
    long = int
    unicode = str


class ImagePreprocessing(Preprocessing):
    """
    ImagePreprocessing is a transformer that transform ImageFeature
    """
    def __init__(self, bigdl_type="float"):
        super(ImagePreprocessing, self).__init__(bigdl_type)

    def __call__(self, image_set, bigdl_type="float"):
        """
        transform ImageSet
        """
        jset = callBigDlFunc(bigdl_type,
                             "transformImageSet", self.value, image_set)
        return ImageSet(jvalue=jset)


class Resize(ImagePreprocessing):
    """
    Resize image
    :param resize_h height after resize
    :param resize_w width after resize
    :param resize_mode if resizeMode = -1, random select a mode from (Imgproc.INTER_LINEAR,
     Imgproc.INTER_CUBIC, Imgproc.INTER_AREA, Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)
    :param use_scale_factor if true, scale factor fx and fy is used, fx = fy = 0
    note that the result of the following are different
    Imgproc.resize(mat, mat, new Size(resizeWH, resizeWH), 0, 0, Imgproc.INTER_LINEAR)
    Imgproc.resize(mat, mat, new Size(resizeWH, resizeWH))
    """
    def __init__(self, resize_h, resize_w, resize_mode=1, use_scale_factor=True,
                 bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgResize", resize_h, resize_w, resize_mode, use_scale_factor)


class Brightness(ImagePreprocessing):
    """
    adjust the image brightness
    :param deltaLow brightness parameter: low bound
    :param deltaHigh brightness parameter: high bound
    """
    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgBrightness", delta_low, delta_high)


class ChannelNormalize(ImagePreprocessing):
    """
    image channel normalize
    :param mean_r mean value in R channel
    :param mean_g mean value in G channel
    :param meanB_b mean value in B channel
    :param std_r std value in R channel
    :param std_g std value in G channel
    :param std_b std value in B channel
    """
    def __init__(self, mean_r, mean_b, mean_g, std_r=1.0, std_g=1.0, std_b=1.0, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgChannelNormalizer", mean_r, mean_g, mean_b, std_r, std_g, std_b)


class MatToTensor(ImagePreprocessing):
    """
    MatToTensor
    """
    def __init__(self, bigdl_type="float"):
        super(MatToTensor, self).__init__(bigdl_type)


class CenterCrop(ImagePreprocessing):
    """
    CenterCrop
    """
    def __init__(self, cropWidth, cropHeight, bigdl_type="float"):
        super(CenterCrop, self).__init__(bigdl_type, cropWidth, cropHeight)


class Hue(ImagePreprocessing):
    """
    adjust the image hue
    :param deltaLow hue parameter: low bound
    :param deltaHigh hue parameter: high bound
    """
    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgHue", delta_low, delta_high)


class Saturation(ImagePreprocessing):
    """
    adjust the image Saturation
    :param deltaLow brightness parameter: low bound
    :param deltaHigh brightness parameter: high bound
    """
    def __init__(self, delta_low, delta_high, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgSaturation", delta_low, delta_high)


class ChannelOrder(ImagePreprocessing):
    """
    random change the channel of an image
    """
    def __init__(self, bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgChannelOrder")


class ColorJitter(ImagePreprocessing):
    """
    Random adjust brightness, contrast, hue, saturation
    :param brightness_prob probability to adjust brightness
    :param brightness_delta brightness parameter
    :param contrast_prob probability to adjust contrast
    :param contrast_lower contrast lower parameter
    :param contrast_upper contrast upper parameter
    :param hue_prob probability to adjust hue
    :param hue_delta hue parameter
    :param saturation_prob probability to adjust saturation
    :param saturation_lower saturation lower parameter
    :param saturation_upper saturation upper parameter
    :param random_order_prob random order for different operation
    :param shuffle  shuffle the transformers
    """
    def __init__(self, brightness_prob=0.5,
                 brightness_delta=32.0,
                 contrast_prob=0.5,
                 contrast_lower=0.5,
                 contrast_upper=1.5,
                 hue_prob=0.5,
                 hue_delta=18.0,
                 saturation_prob=0.5,
                 saturation_lower=0.5,
                 saturation_upper=1.5,
                 random_order_prob=0.0,
                 shuffle=False,
                 bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, "createImgColorJitter",
            brightness_prob, brightness_delta,
            contrast_prob, contrast_lower, contrast_upper,
            hue_prob, hue_delta,
            saturation_prob, saturation_lower, saturation_upper,
            random_order_prob, shuffle)


class AspectScale(ImagePreprocessing):
    """
    Resize the image, keep the aspect ratio. scale according to the short edge
    :param min_size scale size, apply to short edge
    :param scale_multiple_of make the scaled size multiple of some value
    :param max_size max size after scale
    :param resize_mode if resizeMode = -1, random select a mode from
    (Imgproc.INTER_LINEAR, Imgproc.INTER_CUBIC, Imgproc.INTER_AREA,
    Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)
    :param use_scale_factor if true, scale factor fx and fy is used, fx = fy = 0
    :aram min_scale control the minimum scale up for image
    """

    def __init__(self, min_size, scale_multiple_of=1, max_size=1000,
                 resize_mode=1, use_scale_factor=True, min_scale=-1.0,
                 bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "createImgAspectScale",
                                   min_size, scale_multiple_of, max_size,
                                   resize_mode, use_scale_factor, min_scale)


class RandomAspectScale(ImagePreprocessing):
    """
    resize the image by randomly choosing a scale
    :param scales array of scale options that for random choice
    :param scaleMultipleOf Resize test images so that its width and height are multiples of
    :param maxSize Max pixel size of the longest side of a scaled input image
    """
    def __init__(self, scales, scale_multiple_of=1, max_size=1000, bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "createImgRandomAspectScale",
                                   scales, scale_multiple_of, max_size)


class ChannelNormalize(ImagePreprocessing):
    """
    image channel normalize
    :param mean_r mean value in R channel
    :param mean_g mean value in G channel
    :param meanB_b mean value in B channel
    :param std_r std value in R channel
    :param std_g std value in G channel
    :param std_b std value in B channel
    """
    def __init__(self, mean_r, mean_b, mean_g, std_r=1.0, std_g=1.0, std_b=1.0, bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "createImgChannelNormalize",
                                   mean_r, mean_g, mean_b, std_r, std_g, std_b)


class PixelNormalize(ImagePreprocessing):
    """
    Pixel level normalizer, data(i) = data(i) - mean(i)

    :param means pixel level mean, following H * W * C order
    """

    def __init__(self, means, bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "createImgPixelNormalize", means)


class RandomCrop(ImagePreprocessing):
    """
    Random crop a `cropWidth` x `cropHeight` patch from an image.
    The patch size should be less than the image size.

    :param crop_width width after crop
    :param crop_height height after crop
    :param is_clip whether to clip the roi to image boundaries
    """

    def __init__(self, crop_width, crop_height, is_clip=True, bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "createImgRandomCrop",
                                   crop_width, crop_height, is_clip)


class CenterCrop(ImagePreprocessing):
    """
    Crop a `cropWidth` x `cropHeight` patch from center of image.
    The patch size should be less than the image size.
    :param crop_width width after crop
    :param crop_height height after crop
    :param is_clip  clip cropping box boundary
    """

    def __init__(self, crop_width, crop_height, is_clip=True, bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "createImgCenterCrop",
                                   crop_width, crop_height, is_clip)


class FixedCrop(ImagePreprocessing):
    """
    Crop a fixed area of image

    :param x1 start in width
    :param y1 start in height
    :param x2 end in width
    :param y2 end in height
    :param normalized whether args are normalized, i.e. in range [0, 1]
    :param is_clip whether to clip the roi to image boundaries
    """

    def __init__(self, x1, y1, x2, y2, normalized=True, is_clip=True, bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "createImgFixedCrop",
                                   x1, y1, x2, y2, normalized, is_clip)


class Expand(ImagePreprocessing):
    """
    expand image, fill the blank part with the meanR, meanG, meanB

    :param means_r means in R channel
    :param means_g means in G channel
    :param means_b means in B channel
    :param min_expand_ratio min expand ratio
    :param max_expand_ratio max expand ratio
    """

    def __init__(self, means_r=123, means_g=117, means_b=104,
                 min_expand_ratio=1.0,
                 max_expand_ratio=4.0, bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "createImgExpand", means_r, means_g, means_b,
                                   min_expand_ratio, max_expand_ratio)


class Filler(ImagePreprocessing):
    """
    Fill part of image with certain pixel value
    :param start_x start x ratio
    :param start_y start y ratio
    :param end_x end x ratio
    :param end_y end y ratio
    :param value filling value
    """

    def __init__(self, start_x, start_y, end_x, end_y, value=255, bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "createImgFiller",
                                   start_x,
                                   start_y,
                                   end_x,
                                   end_y,
                                   value)


class HFlip(ImagePreprocessing):
    """
    Flip the image horizontally
    """

    def __init__(self, bigdl_type="float"):
        self.value = callBigDlFunc(bigdl_type, "createImgHFlip")
