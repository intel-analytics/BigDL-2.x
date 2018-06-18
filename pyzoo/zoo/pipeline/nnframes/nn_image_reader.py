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

import sys
from bigdl.util.common import callBigDlFunc

if sys.version >= '3':
    long = int
    unicode = str


class NNImageReader:
    """
    Primary DataFrame-based image loading interface, defining API to read images from files
    to DataFrame.
    """

    @staticmethod
    def readImages(path, sc=None, minPartitions=1, resizeH=-1, resizeW=-1, bigdl_type="float"):
        """
        Read the directory of images into DataFrame from the local or remote source.
        :param path Directory to the input data files, the path can be comma separated paths as the
                list of inputs. Wildcards path are supported similarly to sc.binaryFiles(path).
        :param min_partitions A suggestion value of the minimal splitting number for input data.
        :param resizeH height after resize, by default is -1 which will not resize the image
        :param resizeW width after resize, by default is -1 which will not resize the image
        :return DataFrame with a single column "image"; Each record in the column represents
                one image record: Row (uri, height, width, channels, CvType, bytes).
        """
        df = callBigDlFunc(bigdl_type, "nnReadImage", path, sc, minPartitions, resizeH, resizeW)
        df._sc._jsc = sc._jsc
        return df
