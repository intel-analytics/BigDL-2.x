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

import numpy as np
import sys

from bigdl.dataset.dataset import DataSet
from bigdl.transform.vision.image import FeatureTransformer
from bigdl.util.common import get_node_and_core_number
from zoo.common.utils import callZooFunc
from zoo.common import Sample, JTensor
from zoo.common.nncontext import getOrCreateSparkContext
from zoo.feature.common import FeatureSet, SampleToMiniBatch
from zoo.feature.image import ImagePreprocessing, ImageFeatureToSample
from zoo.util import nest

if sys.version >= '3':
    long = int
    unicode = str


def _to_tensor_structure(tensors):
    if isinstance(tensors, tuple):
        tensor_structure = TensorMeta(dtype=tensors[0], shape=tensors[1], name="input0")
    elif isinstance(tensors, list):
        tensor_structure = [TensorMeta(dtype=value[0], shape=value[1],
                                       name="list_input_" + str(idx))
                            for (idx, value) in enumerate(tensors)]
    elif isinstance(tensors, dict):
        tensor_structure = {}
        for key, value in tensors.items():
            tensor_structure[key] = TensorMeta(dtype=value[0], shape=value[1], name=key)
    else:
        raise ValueError("In TFDataset.from_rdd, features and labels should be a tuple, "
                         "a list of tuples or a dict of tuples")
    return tensor_structure


def _tensors_to_rdd(tensors, sc, splits):
    import tensorflow as tf
    if isinstance(tensors, np.ndarray):
        tensors = (tensors,)

    if isinstance(tensors, list):
        for i in range(len(tensors)):
            if tensors[i].dtype == np.dtype("float64"):
                tensors[i] = np.float32(tensors[i])

        data_list = _splits(tensors)
        rdd = sc.parallelize(data_list, splits)
        tensor_structure = [TensorMeta(tf.as_dtype(t.dtype),
                                       shape=t.shape[1:],
                                       name="input_%s" % i)
                            for i, t in enumerate(tensors)]
    else:
        flattened = nest.flatten(tensors)
        for i in range(len(flattened)):
            if flattened[i].dtype == np.dtype("float64"):
                flattened[i] = np.float32(flattened[i])
        data_list = _splits(flattened)
        rdd = sc.parallelize(data_list, splits)
        rdd = rdd.map(lambda x: nest.pack_sequence_as(tensors, x))
        tensor_structure = nest.pack_sequence_as(tensors,
                                                 [TensorMeta(tf.as_dtype(t.dtype),
                                                             shape=t.shape[1:],
                                                             name="input_%s" % i)
                                                  for i, t in enumerate(flattened)])
    return rdd, tensor_structure


def _splits(tensors):
    data_list = []
    data_size = tensors[0].shape[0]
    for i in range(data_size):
        sample = []
        for j in range(len(tensors)):
            sample.append(tensors[j][i])
        data_list.append(sample)
    return data_list


class MergeFeatureLabelImagePreprocessing(ImagePreprocessing):
    def __init__(self, bigdl_type="float"):
        super(MergeFeatureLabelImagePreprocessing, self).__init__(bigdl_type)


class MergeFeatureLabelFeatureTransformer(FeatureTransformer):
    def __init__(self, bigdl_type="float"):
        super(MergeFeatureLabelFeatureTransformer, self).__init__(bigdl_type)


class TensorMeta(object):
    def __init__(self, dtype, name=None, shape=None):
        self.dtype = dtype
        self.name = name
        self.shape = shape


class TFDataset(object):
    def __init__(self, tensor_structure, batch_size,
                 batch_per_thread, hard_code_batch_size=False):
        """

        TFDataset represents a distributed collection of elements (backed by a RDD)
        to be feed into Tensorflow graph.

        :param tensor_structure: a nested structure of TensorMeta objects specifying the
        name, shape and data type of each element in this TFDataset
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        """

        if batch_size > 0 and batch_per_thread > 0:
            raise ValueError("bath_size and batch_per_thread should not be set simultaneously")

        self.has_batch = True
        node_num, core_num = get_node_and_core_number()
        self.total_core_num = node_num * core_num
        if batch_size > 0:
            if batch_size % self.total_core_num != 0:
                raise ValueError("batch_size should be a multiple " +
                                 "of total core number, but got batch_size: " +
                                 "%s where total core number is %s" % (batch_size,
                                                                       self.total_core_num))
        if batch_size <= 0 and batch_per_thread <= 0:
            batch_per_thread = 1
            batch_size = self.total_core_num
            self.has_batch = False

        self.batch_size = batch_size
        self.batch_per_thread = batch_per_thread
        self.hard_code_batch_size = hard_code_batch_size
        self.tensor_structure = tensor_structure

        if not self.hard_code_batch_size:
            self.output_shapes = nest.pack_sequence_as(
                self.tensor_structure, [[None] + list(t.shape)
                                        if t is not None else None
                                        for t in nest.flatten(self.tensor_structure)])
        else:
            if self.batch_per_thread > 0:
                self.output_shapes = nest.pack_sequence_as(
                    self.tensor_structure, [[self.batch_per_thread] + t.shape
                                            if t is not None else None
                                            for t in nest.flatten(self.tensor_structure)])
            else:
                self.output_shapes = nest.pack_sequence_as(
                    self.tensor_structure, [[self.batch_size // self.total_core_num] + t.shape
                                            if t is not None else None
                                            for t in nest.flatten(self.tensor_structure)])

        self.input_names = nest.pack_sequence_as(
            self.tensor_structure, [t.name
                                    if t is not None else None
                                    for t in nest.flatten(self.tensor_structure)])

        self._tensors = None

    def _create_placeholders(self):
        import tensorflow as tf
        if not self.hard_code_batch_size:
            tensors = nest.pack_sequence_as(
                self.tensor_structure, [tf.placeholder(name=t.name,
                                                       dtype=t.dtype,
                                                       shape=[None] + list(t.shape))
                                        for t in nest.flatten(self.tensor_structure)])
        else:
            if self.batch_per_thread > 0:
                tensors = nest.pack_sequence_as(
                    self.tensor_structure,
                    [tf.placeholder(name=t.name,
                                    dtype=t.dtype,
                                    shape=[self.batch_per_thread] + list(t.shape))
                     for t in nest.flatten(self.tensor_structure)])
            else:
                tensors = nest.pack_sequence_as(
                    self.tensor_structure,
                    [tf.placeholder(name=t.name,
                                    dtype=t.dtype,
                                    shape=[self.batch_size // self.total_core_num] + list(t.shape))
                     for t in nest.flatten(self.tensor_structure)])

        for tensor in nest.flatten(tensors):
            tf.get_default_graph().clear_collection(tensor.name)
            tf.add_to_collection(tensor.name, self)

        self._original_tensors = tensors
        self._tensors = tensors

        if not self.has_batch:
            self._tensors = nest.pack_sequence_as(self.tensor_structure,
                                                  [t[0] for t in nest.flatten(tensors)])

        return tensors

    @property
    def tensors(self):
        """
        a nested structure of TensorFlow tensor object in TensorFlow graph.
        The elements of this dataset will be fed into these tensors on each iteration.
        :return: the nested structure of TensorFlow tensor object
        """

        if self._tensors is None:
            self._create_placeholders()

        return self._tensors

    @property
    def feature_tensors(self):

        if self._tensors is None:
            self._create_placeholders()

        if not isinstance(self._tensors, tuple):
            raise ValueError("To use feature_tensors, " +
                             "the element in TFDataset must be a tuple of two components. " +
                             "Please use TFDataset.from_rdd(rdd, features=..., labels=...). ")

        return self._tensors[0]

    @property
    def label_tensors(self):

        if self._tensors is None:
            self._create_placeholders()

        if not isinstance(self._tensors, tuple):
            raise ValueError("To use label_tensors, " +
                             "the element in TFDataset must be a tuple of two components. " +
                             "Please use TFDataset.from_rdd(rdd, features=..., labels=...). ")

        return self._tensors[1]

    @staticmethod
    def _to_tensor_structure(features, labels):
        feature_structure = _to_tensor_structure(features)
        if labels is not None:
            label_structure = _to_tensor_structure(labels)
            tensor_structure = (feature_structure, label_structure)

        else:
            tensor_structure = (feature_structure,)
        return tensor_structure

    def get_prediction_data(self):
        """
        :return: an object that can be used for TFNet.predict
        e.g. an RDD of Sample or a ImageSet
        """
        raise NotImplementedError

    def get_evaluation_data(self):
        """
        :return: an object that can be used for TFNet.evaluate,
        e.g. an RDD of Sample or a ImageSet
        """
        raise NotImplementedError

    def get_training_data(self):
        """
        :return: an object that can be used to create a BigDL optimizer,
        e.g. an RDD of Sample or a DataSet
        """
        raise NotImplementedError

    def get_validation_data(self):
        """
        :return: an object that can be used to set validation in a BigDL optimizer,
        e.g. an RDD of Sample or a DataSet
        """
        raise NotImplementedError

    def get_num_partitions(self):
        """
        :return: the num of partitions of the underlying RDD
        """
        raise NotImplementedError

    def map(self, map_fn):
        return MapDataset(self, map_fn)

    @staticmethod
    def from_rdd(*args, **kwargs):
        """
        Create a TFDataset from a rdd.

        For training and evaluation, both `features` and `labels` arguments should be specified.
        The element of the rdd should be a tuple of two, (features, labels), each has the
        same structure of numpy.ndarrays of the argument `features`, `labels`.

        E.g. if `features` is [(tf.float32, [10]), (tf.float32, [20])],
        and `labels` is {"label1":(tf.float32, [10]), "label2": (tf.float32, [20])}
        then a valid element of the rdd could be

        (
        [np.zeros(dtype=float, shape=(10,), np.zeros(dtype=float, shape=(10,)))],
         {"label1": np.zeros(dtype=float, shape=(10,)),
          "label2":np.zeros(dtype=float, shape=(10,))))}
        )

        If `labels` is not specified,
        then the above element should be changed to
        [np.zeros(dtype=float, shape=(10,), np.zeros(dtype=float, shape=(10,)))]

        For inference, `labels` can be not specified.
        The element of the rdd should be some ndarrays of the same structure of the `features`
        argument.

        A note on the legacy api: if you are using `names`, `shapes`, `types` arguments,
        each element of the rdd should be a list of numpy.ndarray.

        :param rdd: a rdd containing the numpy.ndarrays to be used
        for training/evaluation/inference
        :param features: the structure of input features, should one the following:
               - a tuple (dtype, shape), e.g. (tf.float32, [28, 28, 1])
               - a list of such tuple [(dtype1, shape1), (dtype2, shape2)],
                     e.g. [(tf.float32, [10]), (tf.float32, [20])],
               - a dict of such tuple, mapping string names to tuple {"name": (dtype, shape},
                     e.g. {"input1":(tf.float32, [10]), "input2": (tf.float32, [20])}

        :param labels: the structure of input labels, format is the same as features
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param val_rdd: validation data with the same structure of rdd
        :param sequential_order: whether to iterate the elements in the Dataset
                                 in sequential order when training.
        :param shuffle: whether to shuffle the elements in each partition before each epoch
                        when training
        :return: a TFDataset
        """
        return TFNdarrayDataset.from_rdd(*args, **kwargs)

    @staticmethod
    def from_ndarrays(*args, **kwargs):
        """
        Create a TFDataset from a nested structure of numpy ndarrays. Each element
        in the resulting TFDataset has the same structure of the argument tensors and
        is created by indexing on the first dimension of each ndarray in the tensors
        argument.

        This method is equivalent to sc.parallize the tensors and call TFDataset.from_rdd

        :param tensors: the numpy ndarrays
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param val_tensors: the numpy ndarrays used for validation during training
        :param sequential_order: whether to iterate the elements in the Dataset
                                 in sequential order when training.
        :param shuffle: whether to shuffle the elements in each partition before each epoch
                        when training
        :return: a TFDataset
        """
        return TFNdarrayDataset.from_ndarrays(*args, **kwargs)

    @staticmethod
    def from_image_set(image_set, image, label=None,
                       batch_size=-1, batch_per_thread=-1,
                       hard_code_batch_size=False,
                       validation_image_set=None,
                       sequential_order=False,
                       shuffle=True):
        """
        Create a TFDataset from a ImagetSet. Each ImageFeature in the ImageSet should
        already has the "sample" field, i.e. the result of ImageSetToSample transformer

        :param image_set: the ImageSet used to create this TFDataset
        :param image: a tuple of two, the first element is the type of image, the second element
        is the shape of this element, i.e. (tf.float32, [224, 224, 3]))
        :param label: a tuple of two, the first element is the type of label, the second element
        is the shape of this element, i.e. (tf.int32, [1]))
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_image_set: the ImageSet used for validation during training
        :param sequential_order: whether to iterate the elements in the Dataset
                                 in sequential order when training.
        :param shuffle: whether to shuffle the elements in each partition before each epoch
                        when training
        :return: a TFDataset
        """
        tensor_structure = TFDataset._to_tensor_structure(image, label)
        return TFImageDataset(image_set, tensor_structure, batch_size,
                              batch_per_thread, hard_code_batch_size,
                              validation_image_set,
                              sequential_order=sequential_order, shuffle=shuffle)

    @staticmethod
    def from_text_set(text_set, text, label=None,
                      batch_size=-1, batch_per_thread=-1,
                      hard_code_batch_size=False, validation_image_set=None,
                      sequential_order=False, shuffle=True):
        """
        Create a TFDataset from a TextSet. The TextSet must be transformed to Sample, i.e.
        the result of TextFeatureToSample transformer.
        :param text_set: the TextSet used to create this TFDataset
        :param text: a tuple of two, the first element is the type of this input feature,
        the second element is the shape of this element, i.e. (tf.float32, [10, 100, 4])).
        text can also be nested structure of this tuple of two.
        :param label: a tuple of two, the first element is the type of label, the second element
        is the shape of this element, i.e. (tf.int32, [1])). label can also be nested structure of
        this tuple of two.
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_image_set: The TextSet used for validation during training
        :param sequential_order: whether to iterate the elements in the Dataset
                                 in sequential order when training.
        :param shuffle: whether to shuffle the elements in each partition before each epoch
                        when training
        :return: a TFDataset
        """
        tensor_structure = TFDataset._to_tensor_structure(text, label)
        return TFTextDataset(text_set, tensor_structure, batch_size,
                             batch_per_thread, hard_code_batch_size,
                             validation_image_set,
                             sequential_order=sequential_order, shuffle=shuffle)

    @staticmethod
    def from_tfrecord_file(sc, file_path, batch_size=-1, batch_per_thread=-1,
                           hard_code_batch_size=False, validation_file_path=None,
                           sequential_order=False, shuffle=True):
        """
        Create a TFDataset from tfrecord files.
        :param sc: The SparkContext
        :param file_path: comma seperated tfrecord file(s) path
            structure of tensors. Follows the signature:
            * Args:
                * `example`: a string TensorFlow tensor representing a single record
            * Returns:
                a tuple or dictionary of output tensors and the output tensors must be
                of numeric type
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_file_path: The tfrecord files used for validation
        :param sequential_order: whether to iterate the elements in the Dataset
                                 in sequential order when training.
        :param shuffle: whether to shuffle the elements in each partition before each epoch
                        when training
        :return: a TFDataset
        """
        input_format_class = "org.tensorflow.hadoop.io.TFRecordFileInputFormat"
        key_class = "org.apache.hadoop.io.BytesWritable"
        value_class = "org.apache.hadoop.io.NullWritable"
        bytes_rdd = sc.newAPIHadoopFile(file_path, input_format_class,
                                        keyClass=key_class,
                                        valueClass=value_class)
        bytes_rdd = bytes_rdd.map(lambda record: bytearray(record[0]))
        validation_bytes_rdd = None
        if validation_file_path is not None:
            validation_bytes_rdd = sc.newAPIHadoopFile(validation_file_path,
                                                       input_format_class,
                                                       keyClass=key_class,
                                                       valueClass=value_class)
            validation_bytes_rdd = validation_bytes_rdd.map(lambda record: bytearray(record[0]))

        return TFBytesDataset(bytes_rdd, batch_size, batch_per_thread,
                              hard_code_batch_size, validation_bytes_rdd,
                              sequential_order=sequential_order, shuffle=shuffle)

    @staticmethod
    def from_feature_set(dataset, features, labels=None, batch_size=-1, batch_per_thread=-1,
                         hard_code_batch_size=False, validation_dataset=None):
        """
        Create a TFDataset from a FeatureSet. Currently, the element in this Feature set must be a
        Sample, i.e. the result of ImageFeatureToSample transformer
        :param dataset: the feature set used to create this TFDataset
        :param features: a tuple of two, the first element is the type of this input feature,
        the second element is the shape of this element, i.e. (tf.float32, [224, 224, 3])).
        text can also be nested structure of this tuple of two.
        :param labels: a tuple of two, the first element is the type of label, the second element
        is the shape of this element, i.e. (tf.int32, [1])). label can also be nested structure of
        this tuple of two.
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_dataset: The FeatureSet used for validation during training
        :return: a TFDataset
        """
        tensor_structure = TFDataset._to_tensor_structure(features, labels)

        return TFFeatureDataset(dataset, tensor_structure, batch_size,
                                batch_per_thread, hard_code_batch_size,
                                validation_dataset)

    @staticmethod
    def from_string_rdd(string_rdd, batch_size=-1, batch_per_thread=-1,
                        hard_code_batch_size=False, validation_string_rdd=None):
        """
        Create a TFDataset from a RDD of strings. Each element is the RDD should be a single string.
        The returning TFDataset's feature_tensors has only one Tensor. the type of the Tensor
        is tf.string, and the shape is (None,). The returning don't have label_tensors. If the
        dataset is used for training, the label should be encoded in the string.
        :param string_rdd: the RDD of strings
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_string_rdd: the RDD of strings to be used in validation
        :return: a TFDataset
        """
        string_rdd = string_rdd.map(lambda x: bytearray(x, "utf-8"))
        if validation_string_rdd is not None:
            validation_string_rdd = validation_string_rdd.map(lambda x: bytearray(x, "utf-8"))
        return TFBytesDataset(string_rdd, batch_size, batch_per_thread,
                              hard_code_batch_size, validation_string_rdd)

    @staticmethod
    def from_bytes_rdd(bytes_rdd, batch_size=-1, batch_per_thread=-1,
                       hard_code_batch_size=False, validation_bytes_rdd=None):
        """
        Create a TFDataset from a RDD of bytes. Each element is the RDD should be a bytes object.
        The returning TFDataset's feature_tensors has only one Tensor. the type of the Tensor
        is tf.string, and the shape is (None,). The returning don't have label_tensors. If the
        dataset is used for training, the label should be encoded in the bytes.
        :param bytes_rdd: the RDD of bytes
        :param batch_size: the batch size, used for training, should be a multiple of
        total core num
        :param batch_per_thread: the batch size for each thread, used for inference or evaluation
        :param hard_code_batch_size: whether to hard code the batch_size into tensorflow graph,
        if True, the static size of the first dimension of the resulting tensors is
        batch_size/total_core_num (training) or batch_per_thread for inference; if False,
        it is None.
        :param validation_bytes_rdd: the RDD of bytes to be used in validation
        :return: a TFDataset
        """
        return TFBytesDataset(bytes_rdd, batch_size, batch_per_thread,
                              hard_code_batch_size, validation_bytes_rdd)


class MapDataset(TFDataset):

    def __init__(self, pre_dataset, map_fn):
        self.pre_dataset = pre_dataset
        self.map_fn = map_fn
        super(MapDataset, self).__init__(self.pre_dataset.tensor_structure,
                                         self.pre_dataset.batch_size,
                                         self.pre_dataset.batch_per_thread,
                                         self.pre_dataset.hard_code_batch_size)

    def _create_placeholders(self):
        self._tensors = self.map_fn(self.pre_dataset.tensors)
        self._original_tensors = self.pre_dataset._original_tensors
        return self._tensors

    def get_prediction_data(self):
        """
        :return: an object that can be used for TFNet.predict
        e.g. an RDD of Sample or a ImageSet
        """
        return self.pre_dataset.get_prediction_data()

    def get_evaluation_data(self):
        """
        :return: an object that can be used for TFNet.evaluate,
        e.g. an RDD of Sample or a ImageSet
        """
        return self.pre_dataset.get_evaluation_data()

    def get_training_data(self):
        """
        :return: an object that can be used to create a BigDL optimizer,
        e.g. an RDD of Sample or a DataSet
        """
        return self.pre_dataset.get_training_data()

    def get_validation_data(self):
        """
        :return: an object that can be used to set validation in a BigDL optimizer,
        e.g. an RDD of Sample or a DataSet
        """
        return self.pre_dataset.get_validation_data()

    def get_num_partitions(self):
        """
        :return: the num of partitions of the underlying RDD
        """
        return self.pre_dataset.get_num_partitions()


class TFFeatureDataset(TFDataset):

    def __init__(self, dataset, tensor_structure, batch_size,
                 batch_per_thread, hard_code_batch_size=False, validation_dataset=None):
        super(TFFeatureDataset, self).__init__(tensor_structure, batch_size,
                                               batch_per_thread, hard_code_batch_size)
        self.dataset = dataset
        self.validation_dataset = validation_dataset

    def get_prediction_data(self):
        raise Exception("TFFeatureDataset is only supported in training")

    def get_evaluation_data(self):
        raise Exception("TFFeatureDataset is only supported in training")

    def get_training_data(self):
        fs = self.dataset.transform(MergeFeatureLabelFeatureTransformer())
        fs = fs.transform(SampleToMiniBatch(self.batch_size))
        return fs

    def get_validation_data(self):
        if self.validation_dataset is not None:
            fs = self.validation_dataset.transform(
                MergeFeatureLabelFeatureTransformer())
            fs = fs.transform(SampleToMiniBatch(self.batch_size))
            return fs
        return None

    def get_num_partitions(self):
        raise NotImplementedError()


class TFBytesDataset(TFDataset):

    def get_num_partitions(self):
        self.train_rdd.getNumPartitions()

    def __init__(self, string_rdd, batch_size,
                 batch_per_thread, hard_code_batch_size=False,
                 validation_string_rdd=None, sequential_order=False, shuffle=True):
        import tensorflow as tf
        tensor_structure = (TensorMeta(dtype=tf.string, shape=(), name="input"),)

        super(TFBytesDataset, self).__init__(tensor_structure, batch_size,
                                             batch_per_thread, hard_code_batch_size)

        self.train_rdd = string_rdd
        self.validation_rdd = validation_string_rdd
        self.sequential_order = sequential_order
        self.shuffle = shuffle

    def get_prediction_data(self):
        jvalue = callZooFunc("float", "createMiniBatchRDDFromStringRDD",
                             self.train_rdd,
                             self.batch_per_thread)
        rdd = jvalue.value().toJavaRDD()
        return rdd

    def get_evaluation_data(self):
        raise NotImplementedError()

    def get_training_data(self):
        jvalue = callZooFunc("float", "createMiniBatchRDDFromStringRDD",
                             self.train_rdd,
                             self.batch_size)
        rdd = jvalue.value().toJavaRDD()
        fs = FeatureSet.rdd(rdd,
                            sequential_order=self.sequential_order,
                            shuffle=self.shuffle)
        return fs

    def get_validation_data(self):
        if self.validation_rdd is not None:
            jvalue = callZooFunc("float", "createMiniBatchRDDFromStringRDD",
                                 self.validation_rdd,
                                 self.batch_size)
            rdd = jvalue.value().toJavaRDD()
            fs = FeatureSet.rdd(rdd,
                                sequential_order=self.sequential_order,
                                shuffle=self.shuffle)
            return fs
        return None


class TFTextDataset(TFDataset):

    def __init__(self, text_set, tensor_structure, batch_size,
                 batch_per_thread, hard_code_batch_size=False,
                 validation_text_set=None, sequential_order=False, shuffle=True):
        super(TFTextDataset, self).__init__(tensor_structure, batch_size,
                                            batch_per_thread, hard_code_batch_size)
        self.text_set = text_set
        self.validation_text_set = validation_text_set
        self.sequential_order = sequential_order
        self.shuffle = shuffle

    def get_prediction_data(self):
        rdd = self.text_set.get_samples().map(
            lambda sample: Sample.from_jtensor(features=sample.features,
                                               labels=JTensor.from_ndarray(np.array([0.0]))))
        rdd_wrapper = callZooFunc("float", "zooRDDSampleToMiniBatch", rdd, self.batch_per_thread)
        return rdd_wrapper.value().toJavaRDD()

    def get_evaluation_data(self):
        return self.text_set.get_samples()

    def get_training_data(self):
        sample_rdd = self.text_set.get_samples().map(
            lambda sample: Sample.from_jtensor(features=sample.features + sample.labels,
                                               labels=JTensor.from_ndarray(np.array([0.0]))))
        fs = FeatureSet.sample_rdd(sample_rdd,
                                   sequential_order=self.sequential_order,
                                   shuffle=self.shuffle)
        fs = fs.transform(SampleToMiniBatch(self.batch_size))
        return fs

    def get_validation_data(self):
        if self.validation_text_set is not None:
            sample_rdd = self.validation_text_set.get_samples().map(
                lambda sample: Sample.from_jtensor(features=sample.features + sample.labels,
                                                   labels=JTensor.from_ndarray(np.array([0.0]))))
            fs = FeatureSet.sample_rdd(sample_rdd,
                                       sequential_order=self.sequential_order,
                                       shuffle=self.shuffle)
            fs = fs.transform(SampleToMiniBatch(self.batch_size))
            return fs
        return None

    def get_num_partitions(self):
        return self.text_set.get_samples().getNumPartitions()


class TFImageDataset(TFDataset):
    def __init__(self, image_set, tensor_structure, batch_size,
                 batch_per_thread, hard_code_batch_size=False,
                 validation_image_set=None,
                 sequential_order=False, shuffle=True):
        super(TFImageDataset, self).__init__(tensor_structure, batch_size,
                                             batch_per_thread, hard_code_batch_size)
        self.image_set = image_set
        self.validation_image_set = validation_image_set
        self.sequential_order = sequential_order
        self.shuffle = shuffle

    def get_prediction_data(self):
        return self.image_set

    def get_evaluation_data(self):
        return self.image_set.to_image_frame()

    def get_training_data(self):
        fs = FeatureSet.image_set(self.image_set,
                                  sequential_order=self.sequential_order,
                                  shuffle=self.shuffle)
        fs = fs.transform(MergeFeatureLabelImagePreprocessing())
        fs = fs.transform(ImageFeatureToSample())
        fs = fs.transform(SampleToMiniBatch(self.batch_size))

        return fs

    def get_validation_data(self):
        if self.validation_image_set is not None:
            fs = FeatureSet.image_set(self.validation_image_set,
                                      sequential_order=self.sequential_order,
                                      shuffle=self.shuffle)
            fs = fs.transform(MergeFeatureLabelImagePreprocessing())
            fs = fs.transform(ImageFeatureToSample())
            fs = fs.transform(SampleToMiniBatch(self.batch_size))
            return fs
        return None

    def get_num_partitions(self):
        return self.image_set.get_image().getNumPartitions()


class TFNdarrayDataset(TFDataset):

    def __init__(self, rdd, tensor_structure, batch_size,
                 batch_per_thread, hard_code_batch_size=False,
                 val_rdd=None, sequential_order=True, shuffle=False):

        super(TFNdarrayDataset, self).__init__(tensor_structure, batch_size,
                                               batch_per_thread, hard_code_batch_size)

        self.val_rdd = val_rdd
        self.rdd = rdd
        self.sequential_order = sequential_order
        self.shuffle = shuffle

    def get_prediction_data(self):
        rdd = self.rdd.map(lambda t: Sample.from_ndarray(
            nest.flatten(t[0] if isinstance(t, tuple) else t), np.array([0.0])))
        rdd_wrapper = callZooFunc("float", "zooRDDSampleToMiniBatch", rdd, self.batch_per_thread)
        return rdd_wrapper.value().toJavaRDD()

    def get_evaluation_data(self):
        if isinstance(self.tensor_structure, tuple):
            return self.rdd.map(
                lambda t: Sample.from_ndarray(nest.flatten(t[0]), nest.flatten(t[1])))
        return self.rdd.map(lambda t: Sample.from_ndarray(nest.flatten(t), np.array([0.0])))

    def get_training_data(self):
        sample_rdd = self.rdd.map(
            lambda t: Sample.from_ndarray(nest.flatten(t), np.array([0.0])))
        fs = FeatureSet.sample_rdd(sample_rdd,
                                   sequential_order=self.sequential_order,
                                   shuffle=self.shuffle)
        fs = fs.transform(SampleToMiniBatch(self.batch_size))
        return fs

    def get_validation_data(self):
        if self.val_rdd is not None:
            sample_rdd = self.val_rdd.map(
                lambda t: Sample.from_ndarray(nest.flatten(t), np.array([0.0])))
            fs = FeatureSet.sample_rdd(sample_rdd,
                                       sequential_order=self.sequential_order,
                                       shuffle=self.shuffle)
            fs = fs.transform(SampleToMiniBatch(self.batch_size))
            return fs
        return None

    def get_num_partitions(self):
        return self.rdd.getNumPartitions()

    @staticmethod
    def from_rdd(rdd, names=None, shapes=None, types=None,
                 batch_size=-1, batch_per_thread=-1,
                 hard_code_batch_size=False, val_rdd=None,
                 features=None, labels=None,
                 sequential_order=False,
                 shuffle=True):

        import tensorflow as tf

        if features is not None:
            feature_structure = _to_tensor_structure(features)
            if labels is not None:
                label_structure = _to_tensor_structure(labels)
                tensor_structure = (feature_structure, label_structure)

            else:
                tensor_structure = (feature_structure,)

            return TFNdarrayDataset(rdd, tensor_structure,
                                    batch_size, batch_per_thread,
                                    hard_code_batch_size, val_rdd,
                                    sequential_order=sequential_order,
                                    shuffle=shuffle)

        if names is not None or shapes is not None or types is not None:
            if not names:
                names = ["features", "labels"]
            if not shapes:
                shapes = [None] * len(names)

            if not types:
                types = [tf.float32] * len(names)
            tensor_structure = []
            for i in range(len(names)):
                tensor_structure.append(TensorMeta(types[i], name=names[i], shape=shapes[i]))
        else:
            tensor_structure = [TensorMeta(dtype=tf.float32), TensorMeta(dtype=tf.float32)]

        return TFNdarrayDataset(rdd, tensor_structure,
                                batch_size, batch_per_thread,
                                hard_code_batch_size, val_rdd,
                                sequential_order=sequential_order, shuffle=shuffle)

    @staticmethod
    def from_ndarrays(tensors, batch_size=-1, batch_per_thread=-1,
                      hard_code_batch_size=False, val_tensors=None,
                      sequential_order=False, shuffle=True):
        sc = getOrCreateSparkContext()
        node_num, core_num = get_node_and_core_number()
        total_core_num = node_num * core_num

        rdd, tensor_structure = _tensors_to_rdd(tensors, sc, total_core_num)

        val_rdd = None
        if val_tensors is not None:
            val_rdd, _ = _tensors_to_rdd(val_tensors, sc, total_core_num)

        return TFNdarrayDataset(rdd, tensor_structure, batch_size,
                                batch_per_thread, hard_code_batch_size,
                                val_rdd, sequential_order=sequential_order, shuffle=shuffle)
