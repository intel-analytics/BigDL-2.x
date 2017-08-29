import sys
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from pyspark.mllib.linalg import Vectors

if sys.version >= '3':
    long = int
    unicode = str


class FeatureTransformer(JavaValue):

    def __init__(self, bigdl_type="float", *args):
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), *args)

    def transform(self, image, bigdl_type="float"):
        vector = [Vectors.dense(image), Vectors.dense(image.shape)]
        transformed = callBigDlFunc(bigdl_type, "transform", self.value, vector)
        return transformed[0].array.reshape(transformed[1].array)

    def __call__(self, image_rdd, bigdl_type="float"):
        vector_rdd = image_rdd.map(lambda image: [Vectors.dense(image), Vectors.dense(image.shape)])
        transformed_rdd = callBigDlFunc(bigdl_type, "transformRdd", self.value, vector_rdd)
        return transformed_rdd.map(lambda transformed: transformed[0].array.reshape(transformed[1].array))

class Pipeline(JavaValue):

    def __init__(self, transformers, bigdl_type="float"):
        self.transformer = callBigDlFunc(bigdl_type, "chainTransformer", transformers)

    def transform(self, image, bigdl_type="float"):
        vector = [Vectors.dense(image), Vectors.dense(image.shape)]
        transformed = callBigDlFunc(bigdl_type, "transform", self.transformer, vector)
        return transformed[0].array.reshape(transformed[1].array)

    def __call__(self, image_rdd, bigdl_type="float"):
        vector_rdd = image_rdd.map(lambda image: [Vectors.dense(image), Vectors.dense(image.shape)])
        transformed_rdd = callBigDlFunc(bigdl_type, "transformRdd", self.transformer, vector_rdd)
        return transformed_rdd.map(lambda transformed: transformed[0].array.reshape(transformed[1].array))




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
