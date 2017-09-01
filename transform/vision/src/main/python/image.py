import sys
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import *

if sys.version >= '3':
    long = int
    unicode = str


class FeatureTransformer(JavaValue):

    def __init__(self, bigdl_type="float", *args):
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), *args)

    def transform(self, image_feature, bigdl_type="float"):
        callBigDlFunc(bigdl_type, "transformImageFeature", self.value, image_feature)
        return image_feature

    def __call__(self, image_feature_rdd, bigdl_type="float"):
        return callBigDlFunc(bigdl_type,
                             "transformImageFeatureRdd", self.value, image_feature_rdd)

class Pipeline(JavaValue):

    def __init__(self, transformers, bigdl_type="float"):
        for transfomer in transformers:
            assert transfomer.__class__.__bases__[0].__name__ == "FeatureTransformer", "the transformer should be " \
                                                                                       "subclass of FeatureTransformer"

        self.transformer = callBigDlFunc(bigdl_type, "chainFeatureTransformer", transformers)

    def transform(self, image_feature, bigdl_type="float"):
        callBigDlFunc(bigdl_type, "transformImageFeature", self.transformer, image_feature)
        return image_feature

    def __call__(self, image_feature_rdd, bigdl_type="float"):
        return callBigDlFunc(bigdl_type,
                                         "transformImageFeatureRdd", self.transformer, image_feature_rdd)

class ImageFeature(JavaValue):

    def __init__(self, image=None, label=None, path=None, bigdl_type="float"):
        image_tensor = JTensor.from_ndarray(image) if image is not None else None
        label_tensor = JTensor.from_ndarray(label) if label is not None else None
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), image_tensor, label_tensor, path)

    def to_sample(self, bigdl_type="float"):
        return callBigDlFunc(bigdl_type, "imageFeatureToSample", self.value)


    def get_image(self, bigdl_type="float"):
        tensor = callBigDlFunc(bigdl_type, "getImage", self.value)
        return tensor.to_ndarray()

class ImageFeatureRdd(JavaValue):

    def __init__(self, jvalue, bigdl_type, *args):
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), *args)
        self.bigdl_type = bigdl_type


def ndarray_to_image_feature(ndarray_rdd, bigdl_type="float"):
    tensor_rdd = ndarray_rdd.map(lambda image: [JTensor.from_ndarray(image)])
    return callBigDlFunc(bigdl_type, "tensorRddToImageFeatureRdd", tensor_rdd)


def image_feature_to_sample(image_feature_rdd, bigdl_type="float"):
    return callBigDlFunc(bigdl_type,
                         "imageFeatureRddToSampleRdd", image_feature_rdd)

class Resize(FeatureTransformer):

    def __init__(self, resize_h, resize_w, resize_mode = 1, bigdl_type="float"):
        super(Resize, self).__init__(bigdl_type, resize_h, resize_w, resize_mode)

class Brightness(FeatureTransformer):

    def __init__(self, delta_low, delta_high, bigdl_type="float"):
            super(Brightness, self).__init__(bigdl_type, delta_low, delta_high)

class ChannelOrder(FeatureTransformer):

    def __init__(self, bigdl_type="float"):
            super(ChannelOrder, self).__init__(bigdl_type)

class Contrast(FeatureTransformer):

    def __init__(self, delta_low, delta_high, bigdl_type="float"):
            super(Contrast, self).__init__(bigdl_type, delta_low, delta_high)

class Saturation(FeatureTransformer):

    def __init__(self, delta_low, delta_high, bigdl_type="float"):
            super(Saturation, self).__init__(bigdl_type, delta_low, delta_high)

class Hue(FeatureTransformer):

    def __init__(self, delta_low, delta_high, bigdl_type="float"):
            super(Hue, self).__init__(bigdl_type, delta_low, delta_high)

class Crop(FeatureTransformer):

    def __init__(self, normalized=True, roi=None, roiKey=None, bigdl_type="float"):
        super(Crop, self).__init__(bigdl_type, normalized, roi, roiKey)

class ChannelNormalize(FeatureTransformer):

    def __init__(self, mean_r, mean_b, mean_g, bigdl_type="float"):
        super(ChannelNormalize, self).__init__(bigdl_type, mean_r, mean_g, mean_b)


class RandomCrop(FeatureTransformer):

    def __init__(self, crop_width, crop_height, bigdl_type="float"):
        super(RandomCrop, self).__init__(bigdl_type, crop_width, crop_height)

class CenterCrop(FeatureTransformer):

    def __init__(self, crop_width, crop_height, bigdl_type="float"):
        super(CenterCrop, self).__init__(bigdl_type, crop_width, crop_height)

class HFlip(FeatureTransformer):

    def __init__(self, bigdl_type="float"):
            super(HFlip, self).__init__(bigdl_type)

class Expand(FeatureTransformer):

    def __init__(self, means_r=123, means_g=117, means_b=104,
                 max_expand_ratio=4.0, bigdl_type="float"):
            super(Expand, self).__init__(bigdl_type, means_r, means_g, means_b, max_expand_ratio)

class RandomTransformer(FeatureTransformer):

    def __init__(self, transformer, prob, bigdl_type="float"):
            super(RandomTransformer, self).__init__(bigdl_type, transformer, prob)


class ColorJitter(FeatureTransformer):

    def __init__(self, brightness_prob = 0.5,
                 brightness_delta = 32.0,
                 contrast_prob = 0.5,
                 contrast_lower = 0.5,
                 contrast_upper = 1.5,
                 hue_prob = 0.5,
                 hue_delta = 18.0,
                 saturation_prob = 0.5,
                 saturation_lower = 0.5,
                 saturation_upper = 1.5,
                 random_order_prob = 0.0,
                 shuffle = False,
                 bigdl_type="float"):
        super(ColorJitter, self).__init__(bigdl_type, brightness_prob,
                                          brightness_delta,
                                          contrast_prob,
                                          contrast_lower,
                                          contrast_upper,
                                          hue_prob,
                                          hue_delta,
                                          saturation_prob,
                                          saturation_lower,
                                          saturation_upper,
                                          random_order_prob,
                                          shuffle)

class RandomSampler(FeatureTransformer):

    def __init__(self):
        super(RandomSampler, self).__init__(bigdl_type)

class RoiCrop(FeatureTransformer):

    def __init__(self, bigdl_type="float"):
        super(RoiCrop, self).__init__(bigdl_type)

class RoiExpand(FeatureTransformer):

    def __init__(self, bigdl_type="float"):
        super(RoiExpand, self).__init__(bigdl_type)

class RoiHFlip(FeatureTransformer):

    def __init__(self, bigdl_type="float"):
        super(RoiHFlip, self).__init__(bigdl_type)

class RoiNormalize(FeatureTransformer):

    def __init__(self, bigdl_type="float"):
        super(RoiNormalize, self).__init__(bigdl_type)

class MatToFloats(FeatureTransformer):

    def __init__(self, valid_height=300, valid_width=300,
                 mean_r=-1, mean_g=-1, mean_b=-1, out_key = "floats", bigdl_type="float"):
        super(MatToFloats, self).__init__(bigdl_type, valid_height, valid_width,
                                          mean_r, mean_g, mean_b, out_key)

