
import sys
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import *
from bigdl.transform.vision.image import *

if sys.version >= '3':
    long = int
unicode = str

class UnmodeDetection(FeatureTransformer):
    def __init__(self, bigdl_type="float"):
        super(UnmodeDetection, self).__init__(bigdl_type)


class ImageMeta(FeatureTransformer):
    def __init__(self, class_num, bigdl_type="float"):
        super(ImageMeta, self).__init__(bigdl_type, class_num)


class Visualizer(FeatureTransformer):

    def __init__(self, label_map, thresh = 0.3, encoding = "png",
                 bigdl_type="float"):
        super(Visualizer, self).__init__(bigdl_type, label_map, thresh, encoding)

def read_coco_label_map():
    """
    load coco label map
    """
    return callBigDlFunc("float", "readCocoLabelMap")

def share_memory(model):
    return callBigDlFunc("float", "shareMemory", model)