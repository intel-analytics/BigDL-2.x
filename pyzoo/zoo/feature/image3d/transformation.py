#
# Copyright 2016 The BigDL Authors.
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
import sys
import numpy as np

from bigdl.util.common import *
from zoo.feature.image.imagePreprocessing import *
from zoo.feature.image.imageset import *

if sys.version >= '3':
    long = int
    unicode = str

# class LocalImageSet3D(LocalImageSet):
#     """
#     LocalImageSet wraps a list of ImageFeature
#     """
#     def __init__(self, image_list=None, label_list=None, jvalue=None, bigdl_type="float"):
#         assert jvalue or image_list, "jvalue and image_list cannot be None in the same time"
#         if jvalue:
#             self.value = jvalue
#         else:
#             # init from image ndarray list and label rdd(optional)
#             image_tensor_list = list(map(lambda image: JTensor.from_ndarray(image), image_list))
#             label_tensor_list = list(map(lambda label: JTensor.from_ndarray(label), label_list))\
#                 if label_list else None
#             self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
#                                        image_tensor_list, label_tensor_list)
#
#         self.bigdl_type = bigdl_type
#
#     def get_image(self, tensor_key="imageTensor", to_chw=True):
#         """
#         get image list from ImageSet
#         """
#         tensors = callBigDlFunc(self.bigdl_type, "localImageSet3DToImageTensor",
#                                 self.value, tensor_key, to_chw)
#         return list(map(lambda tensor: tensor.to_ndarray(), tensors))
#
#
#
# class DistributedImageSet3D(DistributedImageSet):
#     """
#     DistributedImageSet wraps an RDD of ImageFeature
#     """
#
#     def __init__(self, image_rdd=None, label_rdd=None, jvalue=None, bigdl_type="float"):
#         assert jvalue or image_rdd, "jvalue and image_rdd cannot be None in the same time"
#         if jvalue:
#             self.value = jvalue
#         else:
#             # init from image ndarray rdd and label rdd(optional)
#             image_tensor_rdd = image_rdd.map(lambda image: JTensor.from_ndarray(image))
#             label_tensor_rdd = label_rdd.map(lambda label: JTensor.from_ndarray(label))\
#                 if label_rdd else None
#             self.value = callBigDlFunc(bigdl_type, JavaValue.jvm_class_constructor(self),
#                                        image_tensor_rdd, label_tensor_rdd)
#
#         self.bigdl_type = bigdl_type
#
#     def get_image(self, tensor_key="imageTensor", to_chw=True):
#         """
#         get image rdd from ImageSet
#         """
#         tensor_rdd = callBigDlFunc(self.bigdl_type, "distributedImageSet3DToImageTensorRdd",
#                                    self.value, tensor_key, to_chw)
#         return tensor_rdd.map(lambda tensor: tensor.to_ndarray())


class ImagePreprocessing3D(ImagePreprocessing):
    """
    ImagePreprocessing3D is a transformer that transform ImageFeature for 3D image
    """
    def __init__(self, bigdl_type="float", *args):
        super(ImagePreprocessing3D, self).__init__(bigdl_type, *args)


class Crop3D(ImagePreprocessing3D):

    def __init__(self, start, patch_size, bigdl_type="float"):
        super(Crop3D, self).__init__(bigdl_type, start, patch_size)


class RandomCrop3D(ImagePreprocessing3D):

    def __init__(self, crop_depth, crop_height, crop_width, bigdl_type="float"):
        super(RandomCrop3D, self).__init__(bigdl_type, crop_depth, crop_height, crop_width)


class CenterCrop3D(ImagePreprocessing3D):

    def __init__(self, crop_depth, crop_height, crop_width, bigdl_type="float"):
        super(CenterCrop3D, self).__init__(bigdl_type, crop_depth, crop_height, crop_width)


class Rotate3D(ImagePreprocessing3D):

    def __init__(self, rotationAngles, bigdl_type="float"):
        super(Rotate3D, self).__init__(bigdl_type, rotationAngles)


class AffineTransform3D(ImagePreprocessing3D):

    def __init__(self, affine_mat, translation=JTensor.from_ndarray(np.zeros(3)), clamp_mode="clamp", pad_val=0.0, bigdl_type="float"):
        super(AffineTransform3D, self).__init__(bigdl_type, affine_mat, translation, clamp_mode, pad_val)