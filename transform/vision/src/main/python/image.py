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

    def __call__(self, image_frame, bigdl_type="float"):
        jframe = callBigDlFunc(bigdl_type,
                             "transformImageFrame", self.value, image_frame)
        return ImageFrame(jvalue=jframe)

class Pipeline(JavaValue):

    def __init__(self, transformers, bigdl_type="float"):
        for transfomer in transformers:
            assert transfomer.__class__.__bases__[0].__name__ == "FeatureTransformer", "the transformer should be " \
                                                                                       "subclass of FeatureTransformer"

        self.transformer = callBigDlFunc(bigdl_type, "chainedFeatureTransformer", transformers)

    def transform(self, image_feature, bigdl_type="float"):
        callBigDlFunc(bigdl_type, "transformImageFeature", self.transformer, image_feature)
        return image_feature

    def __call__(self, image_frame, bigdl_type="float"):
        jframe = callBigDlFunc(bigdl_type,
                               "transformImageFrame", self.transformer, image_frame)
        return ImageFrame(jvalue=jframe)

class ImageFeature(JavaValue):

    def __init__(self, image=None, label=None, path=None, bigdl_type="float"):
        image_tensor = JTensor.from_ndarray(image) if image is not None else None
        label_tensor = JTensor.from_ndarray(label) if label is not None else None
        self.bigdl_type = bigdl_type
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), image_tensor, label_tensor, path)

    def to_sample(self, float_key="floats", to_chw=True, with_im_info=False):
        return callBigDlFunc(self.bigdl_type, "imageFeatureToSample", self.value, float_key, to_chw, with_im_info)

    def get_image(self, float_key="floats", to_chw=True):
        tensor = callBigDlFunc(self.bigdl_type, "imageFeatureToImageTensor", self.value,
                               float_key, to_chw)
        return tensor.to_ndarray()

    def get_label(self):
        label = callBigDlFunc(self.bigdl_type, "imageFeatureToLabelTensor", self.value)
        return label.to_ndarray()

    def keys(self):
        return callBigDlFunc(self.bigdl_type, "imageFeatureGetKeys", self.value)


class ImageFrame(JavaValue):
    """
    ImageFrame is an RDD of ImageFeature
    """

    def __init__(self, image_rdd=None, label_rdd=None, jvalue=None, bigdl_type="float"):
        assert jvalue or image_rdd, "jvalue and image_rdd cannot be None in the same time"
        if jvalue:
            self.value = jvalue
        else:
            # init from image ndarray rdd and label rdd(optional)
            image_tensor_rdd = image_rdd.map(lambda image: JTensor.from_ndarray(image))
            label_tensor_rdd = label_rdd.map(lambda label: JTensor.from_ndarray(label)) if label_rdd else None
            self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
                                       image_tensor_rdd, label_tensor_rdd)

        self.bigdl_type = bigdl_type

    def get_image(self, float_key="floats", to_chw=True):
        """
        get image rdd from ImageFrame
        """
        tensor_rdd = callBigDlFunc(self.bigdl_type,
                               "imageFrameToImageTensorRdd", self.value, float_key, to_chw)
        return tensor_rdd.map(lambda tensor: tensor.to_ndarray())

    def get_label(self):
        """
        get label rdd from ImageFrame
        """
        tensor_rdd = callBigDlFunc(self.bigdl_type, "imageFrameToLabelTensorRdd", self.value)
        return tensor_rdd.map(lambda tensor: tensor.to_ndarray())

    def to_sample(self, float_key="floats", to_chw=True, with_im_info=False):
        """
        to sample rdd
        """
        return callBigDlFunc(self.bigdl_type,
                             "imageFrameToSampleRdd", self.value, float_key, to_chw, with_im_info)

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

    def __init__(self, mean_r, mean_b, mean_g, std_r, std_g, std_b, bigdl_type="float"):
        super(ChannelNormalize, self).__init__(bigdl_type, mean_r, mean_g, mean_b, std_r, std_g, std_b)


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

    def __init__(self, normalized=True, bigdl_type="float"):
        super(RoiHFlip, self).__init__(bigdl_type, normalized)

class RoiNormalize(FeatureTransformer):

    def __init__(self, bigdl_type="float"):
        super(RoiNormalize, self).__init__(bigdl_type)

class MatToFloats(FeatureTransformer):

    def __init__(self, valid_height=300, valid_width=300,
                 mean_r=-1.0, mean_g=-1.0, mean_b=-1.0, out_key = "floats", bigdl_type="float"):
        super(MatToFloats, self).__init__(bigdl_type, valid_height, valid_width,
                                          mean_r, mean_g, mean_b, out_key)
class AspectScale(FeatureTransformer):

    def __init__(self, scale, scale_multiple_of = 1, max_size = 1000, bigdl_type="float"):
        super(AspectScale, self).__init__(bigdl_type, scale, scale_multiple_of, max_size)


