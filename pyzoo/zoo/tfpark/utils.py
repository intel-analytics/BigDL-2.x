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

from zoo.tfpark.tfnet import TFNet
from zoo.tfpark.tf_optimizer import BigDLMetric, TFModel
from zoo.pipeline.api.keras import metrics as zmetrics
from zoo.tfpark.tf_dataset import TFNdarrayDataset, TFDataset


def to_bigdl_metric(metric):
    metric = metric.lower()
    if metric == "accuracy" or metric == "acc":
        return zmetrics.Accuracy()
    elif metric == "top5accuracy" or metric == "top5acc":
        return zmetrics.Top5Accuracy()
    elif metric == "mae":
        from bigdl.optim.optimizer import MAE
        return MAE()
    elif metric == "auc":
        return zmetrics.AUC()
    elif metric == "treennaccuracy":
        from bigdl.optim.optimizer import TreeNNAccuracy
        return TreeNNAccuracy()
    else:
        raise TypeError("Unsupported metric: %s" % metric)


def evaluate_string_metrics(*,
                            sess,
                            string_metrics,
                            dataset,
                            inputs,
                            targets=None,
                            outputs=None,
                            loss=None,
                            ):

    metrics = {}
    for i, metric in enumerate(string_metrics):
        if metric == "loss":
            assert loss is not None, "loss tensor should not be None if one of the metrics is loss"
            metrics["loss"] = loss
        else:
            assert outputs is not None, "outputs should not be None if non loss metrics exists"
            assert targets is not None, "targets should not be None if non loss metrics exists"

            method = to_bigdl_metric(metric)
            metrics[metric] = BigDLMetric(method,
                                          outputs,
                                          targets)
    result = evaluate_metrics(inputs, sess, dataset, metrics)
    return result


def evaluate_metrics(inputs, sess, dataset, metrics):
    import tensorflow as tf
    if dataset.batch_per_thread > 0:
        batch_size = dataset.batch_per_thread * dataset.get_num_partitions()
    else:
        batch_size = dataset.batch_size

    real_batch_size = tf.shape(inputs[0])[0]

    outputs, eval_methods = TFModel._process_metrics(inputs[0].graph,
                                                     metrics=metrics,
                                                     real_batch_size=real_batch_size)

    tfnet = TFNet.from_session(sess, inputs=inputs, outputs=outputs)

    results = tfnet.evaluate(dataset, batch_size, eval_methods)
    final_result = dict([(r.method, r.result) for r in results])
    return final_result


def _standarize_feature_label_dataset(dataset, model):
    input_names = model.input_names
    output_names = model.output_names

    def _process_labels(ys):
        if isinstance(ys, dict):
            return {k: np.expand_dims(y, axis=-1) if y.ndim == 0 else y for k, y in ys.items()}
        elif isinstance(ys, list):
            return [np.expand_dims(y, axis=-1) if y.ndim == 0 else y for y in ys]
        elif isinstance(ys, tuple):
            return tuple([np.expand_dims(y, axis=-1) if y.ndim == 0 else y for y in ys])
        else:
            return np.expand_dims(ys, axis=-1) if ys.ndim == 0 else ys

    def _training_reorder(x, input_names, output_names):
        assert isinstance(x, tuple)

        return (_reorder(x[0], input_names), _reorder(x[1], output_names))

    def _reorder(x, names):
        if isinstance(x, dict):
            return [x[name] for name in names]
        elif isinstance(x, list) or isinstance(x, tuple):
            return x
        else:
            return [x]

    rdd = dataset.rdd.map(lambda x: (x[0], _process_labels(x[1])))\
        .map(lambda sample: _training_reorder(sample, input_names, output_names))
    if dataset.val_rdd is not None:
        val_rdd = dataset.val_rdd.map(lambda x: (x[0], _process_labels(x[1])))\
            .map(lambda sample: _training_reorder(sample, input_names, output_names))
    else:
        val_rdd = None
    tensor_structure = _training_reorder(dataset.tensor_structure, input_names, output_names)
    new_dataset = TFNdarrayDataset(rdd, tensor_structure, dataset.batch_size,
                                   -1, dataset.hard_code_batch_size, val_rdd)
    new_dataset.batch_per_thread = dataset.batch_per_thread
    return new_dataset


def _standarize_feature_dataset(dataset, model):
    input_names = model.input_names

    def _reorder(x, names):
        if isinstance(x, dict):
            return [x[name] for name in names]
        elif isinstance(x, list):
            return x
        elif isinstance(x, tuple):
            return list(x)
        return [x]

    rdd = dataset.rdd.map(lambda sample: _reorder(sample, input_names))
    feature_schema = _reorder(dataset.tensor_structure[0], input_names)

    dataset = TFNdarrayDataset(rdd, feature_schema, dataset.batch_size,
                               -1, dataset.hard_code_batch_size)
    return dataset
