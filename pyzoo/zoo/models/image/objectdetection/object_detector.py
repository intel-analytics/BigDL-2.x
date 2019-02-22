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
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc

from zoo.models.image.common.image_model import ImageModel
from zoo.feature.image.imageset import *
from zoo.feature.image.imagePreprocessing import *
from bigdl.nn.criterion import Criterion
from bigdl.nn.layer import *
from bigdl.nn.initialization_method import *

if sys.version >= '3':
    long = int
    unicode = str


def read_pascal_label_map():
    """
    load pascal label map
    """
    return callBigDlFunc("float", "readPascalLabelMap")


def read_coco_label_map():
    """
    load coco label map
    """
    return callBigDlFunc("float", "readCocoLabelMap")


def add_conv_relu(prev_nodes, n_input_plane, n_output_plane, kernel, stride, pad,
                name, prefix="conv", n_group=1, propagate_back=True):
    conv = SpatialConvolution(n_input_plane, n_output_plane, kernel, kernel, stride, stride,
                              pad, pad, n_group, propagate_back) \
        .set_init_method(weight_init_method=Xavier(), bias_init_method=Zeros()) \
        .set_name(prefix + name)(prev_nodes)
    relu = ReLU(True).set_name("relu" + name)(conv)
    return relu


class ObjectDetector(ImageModel):
    """
    A pre-trained object detector model.

    :param model_path The path containing the pre-trained model
    """
    def __init__(self, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        super(ObjectDetector, self).__init__(None, bigdl_type)

    @staticmethod
    def load_model(path, weight_path=None, bigdl_type="float"):
        """
        Load an existing object detection model (with weights).

        # Arguments
        path: The path to save the model. Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadObjectDetector", path, weight_path)
        model = ImageModel._do_load(jmodel, bigdl_type)
        model.__class__ = ObjectDetector
        return model


class ImInfo(ImagePreprocessing):
    """
    Generate imInfo
    imInfo is a tensor that contains height, width, scaleInHeight, scaleInWidth
    """
    def __init__(self, bigdl_type="float"):
        super(ImInfo, self).__init__(bigdl_type)


class DummyGT(ImagePreprocessing):
    """

    """
    def __init__(self, bigdl_type="float"):
        super(DummyGT, self).__init__(bigdl_type)


class DecodeOutput(ImagePreprocessing):
    """
    Decode the detection output
    The output of the model prediction is a 1-dim tensor
    The first element of tensor is the number(K) of objects detected,
    followed by [label score x1 y1 x2 y2] X K
    For example, if there are 2 detected objects, then K = 2, the tensor may
    looks like
    ```2, 1, 0.5, 10, 20, 50, 80, 3, 0.3, 20, 10, 40, 70```
    After decoding, it returns a 2-dim tensor, each row represents a detected object
    ```
    1, 0.5, 10, 20, 50, 80
    3, 0.3, 20, 10, 40, 70
    ```
    """
    def __init__(self, bigdl_type="float"):
        super(DecodeOutput, self).__init__(bigdl_type)


class ScaleDetection(ImagePreprocessing):
    """
    If the detection is normalized, for example, ssd detected bounding box is in [0, 1],
    need to scale the bbox according to the original image size.
    Note that in this transformer, the tensor from model output will be decoded,
    just like `DecodeOutput`
    """
    def __init__(self, bigdl_type="float"):
        super(ScaleDetection, self).__init__(bigdl_type)


class Visualizer(ImagePreprocessing):
    """
    Visualizer is a transformer to visualize the detection results
    (tensors that encodes label, score, boundingbox)
    You can call image_frame.get_image() to get the visualized results
    """
    def __init__(self, label_map, thresh=0.3, encoding="png",
                 bigdl_type="float"):
        self.value = callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), label_map, thresh, encoding)

    def __call__(self, image_set, bigdl_type="float"):
        """
        transform ImageSet
        """
        jset = callBigDlFunc(bigdl_type,
                             "transformImageSet", self.value, image_set)
        return ImageSet(jvalue=jset)


class RoiRecordToFeature(Preprocessing):
    """
    Convert ROI image record to ImageFeature.
    """
    def __init__(self, convert_label=False, out_key="bytes", bigdl_type="float"):
        super(RoiRecordToFeature, self).__init__(bigdl_type, convert_label, out_key)


class RoiImageToSSDBatch(Preprocessing):
    """
    Convert a batch of labeled BGR images into a Mini-batch.

    Notice: The totalBatch means a total batch size. In distributed environment, the batch should be
    divided by total core number
    """
    def __init__(self, total_batch, convert_label=True, partition_num=None,
                 keep_image_feature=True, input_key="floats", bigdl_type="float"):
        super(RoiImageToSSDBatch, self).__init__(bigdl_type, total_batch, convert_label,
                                              partition_num, keep_image_feature, input_key)


class MeanAveragePrecision(JavaValue):
    """
    Caculate the percentage that output's max probability index equals target.

    >>> top1 = MeanAveragePrecision(False, ["dog","cat"], True)
    creating: createMeanAveragePrecision
    """
    def __init__(self, use_07_metric, classes, normalized=True, bigdl_type="float"):
        JavaValue.__init__(self, None, bigdl_type, use_07_metric, normalized, classes)


class MultiBoxLossParam():
    def __init__(self, loc_weight=1.0, n_classes=21, share_location=True, overlap_threshold=0.5,
                 bg_label_ind=0, use_difficult_gt=True, neg_pos_ratio=3.0, neg_overlap=0.5):
        self.loc_weight = loc_weight
        self.n_classes = n_classes
        self.share_location = share_location
        self.overlap_threshold = overlap_threshold
        self.bg_label_ind = bg_label_ind
        self.use_difficult_gt = use_difficult_gt
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_overlap = neg_overlap

    def __reduce__(self):
        return MultiBoxLossParam, (self.loc_weight, self.n_classes, self.share_location,
                                   self.overlap_threshold, self.bg_label_ind,
                                   self.use_difficult_gt,self.neg_pos_ratio, self.neg_overlap)

    def __str__(self):
        return "MultiBoxLossParam {loc_weight: %s, n_classes: %s, share_location: %s, " \
               "overlap_threshold: %s, bg_label_ind: %s, use_difficult_gt: %s, " \
                "neg_pos_ratio: %s, neg_overlap: %s}" \
               % (self.loc_weight, self.n_classes, self.share_location,
                  self.overlap_threshold, self.bg_label_ind, self.use_difficult_gt,
                  self.neg_pos_ratio, self.neg_overlap)


class MultiBoxLoss(Criterion):
    """
    MultiBox Loss

    >>> criterion = MultiBoxLoss()
    creating: createMultiBoxLoss
    """
    def __init__(self, multibox_loss_param, bigdl_type="float"):
        super(MultiBoxLoss, self).__init__(None, bigdl_type, multibox_loss_param)


class PreProcessParam():
    def __init__(self, batch_size= 1, scales=[600], scale_multiple_of=1,
                 pixel_mean_rgb=(122.7717, 115.9465, 102.9801),
                 has_label=False, n_partition=-1, norms=(1, 1, 1)):
        self.batch_size = batch_size
        self.scales = scales
        self.scale_multiple_of = scale_multiple_of
        self.pixel_mean_rgb = pixel_mean_rgb
        self.has_label = has_label
        self.n_partition = n_partition
        self.norms = norms

    def __reduce__(self):
        return PreProcessParam, (self.batch_size, self.scales,
                                 self.scale_multiple_of, self.pixel_mean_rgb,
                                 self.has_label, self.n_partition, self.norms)

    def __str__(self):
        return "PreProcessParam {batch_size: %s, scales: %s, " \
               "scale_multiple_of: %s, pixel_mean_rgb: %s, thresh: %s}" \
               % (self.n_classes, self.bbox_vote,
                  self.nms_thresh, self.max_per_image, self.has_label,
                  self.n_partition, self.norms)


class PostProcessParam():
    def __init__(self, n_classes, bbox_vote, nms_thresh=0.3, max_per_image=100, thresh=0.05):
        self.n_classes = n_classes
        self.bbox_vote = bbox_vote
        self.nms_thresh = nms_thresh
        self.max_per_image = max_per_image
        self.thresh = thresh

    def __reduce__(self):
        return PostProcessParam, (self.n_classes, self.bbox_vote,
                                   self.nms_thresh, self.max_per_image,
                                   self.thresh)

    def __str__(self):
        return "PostProcessParam {n_classes: %s, bbox_vote: %s, " \
               "nms_thresh: %s, max_per_image: %s, thresh: %s}" \
               % (self.n_classes, self.bbox_vote,
                  self.nms_thresh, self.max_per_image, self.thresh)


def load_model_weights(src_model, target_model, match_all=True):
    """
    Load weights from pretrained model
    """
    return callBigDlFunc("float", "loadModelWeights", src_model, target_model, match_all)


def load_roi_seq_files(url, sc, partition_num=-1):
    """
    Load roi sequence files to image frame
    """
    return callBigDlFunc("float", "loadRoiSeqFiles", url, sc, partition_num)