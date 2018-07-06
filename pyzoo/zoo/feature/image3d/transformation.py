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

if sys.version >= '3':
    long = int
    unicode = str


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