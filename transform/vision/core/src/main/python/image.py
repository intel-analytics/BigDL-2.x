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
        self.bigdl_type = bigdl_type
        self.transformers = [self.value]

    def transform(self, image, bigdl_type="float"):
        callBigDlFunc(bigdl_type, "chainTransformer", self.transformers)
        vector = [Vectors.dense(image), Vectors.dense(image.shape)]
        transformed = callBigDlFunc(bigdl_type, "transform", self.value, vector)
        return transformed[0].array.reshape(transformed[1].array)

    def transform_rdd(self, image_rdd, bigdl_type="float"):
        vector_rdd = image_rdd.map(lambda image: [Vectors.dense(image), Vectors.dense(image.shape)])
        transformed_rdd = callBigDlFunc(bigdl_type, "transformRdd", self.value, vector_rdd)
        return transformed_rdd.map(lambda transformed: transformed[0].array.reshape(transformed[1].array))

    def __add__(self, other):
        self.transformers.append(other.value)
        return self



class Resize(FeatureTransformer):

    def __init__(self, resize_h, resize_w, resize_mode, bigdl_type="float"):
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

    def __init__(self, bigdl_type="float"):
            super(Crop, self).__init__(bigdl_type)

class HFlip(FeatureTransformer):

    def __init__(self, bigdl_type="float"):
            super(HFlip, self).__init__(bigdl_type)

class Expand(FeatureTransformer):

    def __init__(self, means_r, means_g, means_b, max_expand_ratio, bigdl_type="float"):
            super(Expand, self).__init__(bigdl_type, means_r, means_g, means_b, max_expand_ratio)

class RandomOp(FeatureTransformer):

    def __init__(self, transformer, prob, bigdl_type="float"):
            super(RandomOp, self).__init__(bigdl_type, transformer, prob)

class ColorJitter(FeatureTransformer):

    def __init__(self, bigdl_type="float"):
        super(ColorJitter, self).__init__(bigdl_type)
