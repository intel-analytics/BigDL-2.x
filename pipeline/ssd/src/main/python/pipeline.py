import sys
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import *
from bigdl.nn.layer import *

if sys.version >= '3':
    long = int
    unicode = str

def share_memory(model):
    jmodel = callBigDlFunc("float", "shareMemory", model)
    return Layer.of(jmodel)

def object_detect(model, image_batch_rdd):
    tensor_rdd = callBigDlFunc("float", "objectDetect", model, image_batch_rdd)
    return tensor_rdd.map(lambda tensor: tensor.to_ndarray())

def to_ssd_batch(image_frame, n_partition, batch_per_partition=1):
    return callBigDlFunc("float", "toSsdBatch", image_frame, batch_per_partition * n_partition, n_partition)

def to_frcnn_batch(image_frame, n_partition):
    return callBigDlFunc("float", "toFrcnnBatch", image_frame, n_partition, n_partition)
