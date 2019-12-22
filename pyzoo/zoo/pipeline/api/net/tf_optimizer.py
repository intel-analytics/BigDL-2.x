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

import tempfile
import warnings
import os
import json
import sys

import logging

from bigdl.nn.criterion import Criterion
from bigdl.nn.layer import Layer
from bigdl.util.common import to_list, JavaValue
from zoo.common.utils import callZooFunc
from bigdl.optim.optimizer import MaxEpoch, EveryEpoch
from zoo.pipeline.api.keras.engine.topology import to_bigdl_metric, Loss
from zoo.pipeline.api.net.tf_dataset import MapDataset
from zoo.pipeline.api.net.utils import _find_placeholders, to_bigdl_optim_method
from zoo.pipeline.estimator import Estimator
from zoo.util import nest

if sys.version >= '3':
    long = int
    unicode = str


class IdentityCriterion(Criterion):
    def __init__(self):
        super(IdentityCriterion, self).__init__(None, "float")


class TFValidationMethod(JavaValue):
    def __init__(self, val_method, name, output_indices, label_indices):
        JavaValue.__init__(self, None, "float",
                           val_method, name, output_indices, label_indices)


class StatelessMetric(JavaValue):
    def __init__(self, metric_name, idx):
        self.name = metric_name
        self.idx = idx
        JavaValue.__init__(self, None, "float", metric_name, idx)


class BigDLMetric(object):
    def __init__(self, val_method, outputs, labels):
        self.val_method = val_method
        self.outputs = outputs
        self.labels = labels


class TFTrainingHelper(Layer):
    def __init__(self, path, config_proto, saver, meta, sess):
        self.saver = saver
        self.meta = meta
        self.export_dir = path
        self.sess = sess

        if config_proto is not None:
            import tensorflow as tf
            assert isinstance(config_proto, tf.ConfigProto), \
                "session_config should be a tf.ConfigProto"
            config_proto.use_per_session_threads = True
            byte_arr = bytearray(config_proto.SerializeToString())
        else:
            byte_arr = None

        super(TFTrainingHelper, self).__init__(None, "float", path, byte_arr)

    def save_checkpoint(self):
        callZooFunc(self.bigdl_type, "saveCheckpoint",
                    self.value)

    def get_weights_to_python(self):
        self.save_checkpoint()
        self.saver.restore(self.sess, os.path.join(self.export_dir, "model"))


def _to_operation_name(name):
    return name.split(":")[0]


def _to_floats(vs):
    return [float(v) for v in vs]


class TFModel(object):
    def __init__(self, training_helper_layer, criterion, val_methods):

        self.training_helper_layer = training_helper_layer
        self.criterion = criterion
        self.val_methods = val_methods

    @staticmethod
    def _expand_inputs(inputs, tensors_with_value, loss):
        additional_inputs = []
        additional_values = []
        all_required_inputs = _find_placeholders([loss])
        all_required_inputs_names = [v.name for v in all_required_inputs]
        if tensors_with_value:
            for t, v in tensors_with_value.items():
                if t.name in all_required_inputs_names:
                    additional_inputs.append(t)
                    additional_values.append(v)

        if not isinstance(inputs, list):
            inputs = nest.flatten(inputs)

        inputs = inputs + additional_inputs

        return inputs, additional_values

    @staticmethod
    def _process_session_config(session_config):
        if session_config is not None:
            import tensorflow as tf
            assert isinstance(session_config, tf.ConfigProto), \
                "session_config should be a tf.ConfigProto"
            session_config.use_per_session_threads = True
        return session_config

    @staticmethod
    def _process_grads(graph, grads):

        with graph.as_default():
            from zoo.util.tf import process_grad
            grads = [process_grad(grad) for grad in grads]
        return grads

    @staticmethod
    def _process_metrics(graph, metrics):
        import tensorflow as tf
        outputs = []
        val_methods = None
        if metrics is not None:
            idx = 0
            val_methods = []
            for metric_name in metrics:
                metric = metrics[metric_name]
                if tf.is_numeric_tensor(metric):
                    outputs.append(metric)
                    val_methods.append(StatelessMetric(metric_name, idx))
                    idx += 1
                else:
                    outputs += metric.outputs
                    with graph.as_default():
                        val_labels = [tf.identity(v) for v in metric.labels]
                    outputs += val_labels
                    method = TFValidationMethod(metric.val_method,
                                                metric_name,
                                                list(range(idx, idx + len(metric.outputs))),
                                                list(range(idx + len(metric.outputs),
                                                           idx + len(metric.outputs)
                                                           + len(val_labels))))
                    val_methods.append(method)
                    idx += len(metric.outputs) + len(val_labels)

        outputs = [tf.to_float(output) for output in outputs]
        return outputs, val_methods

    @staticmethod
    def _process_variables_for_unfreeze(graph, variables, updates):
        import tensorflow as tf

        all_trainable_variables = variables

        name2idx = dict([(v.name, idx) for idx, v in enumerate(all_trainable_variables)])

        all_variables = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        update_ops = graph.get_collection(tf.GraphKeys.UPDATE_OPS)

        if updates is not None:
            update_ops += updates

        trainable_variables = [0] * len(all_trainable_variables)
        trainable_assigns = [0] * len(all_trainable_variables)
        trainable_variable_placeholders = [0] * len(all_trainable_variables)
        extra_variables = []
        extra_variable_assigns = []
        extra_variable_assign_placeholders = []
        for v in all_variables:
            p = tf.placeholder(dtype=v.dtype, shape=v.shape)
            a = tf.assign(v, p)

            # special treatment for ResourceVariable
            if v.op.type == "VarHandleOp":
                v_float_value = tf.to_float(v.read_value())
            else:
                v_float_value = tf.to_float(v)

            if v.name in name2idx:
                trainable_variables[name2idx[v.name]] = v_float_value
                trainable_assigns[name2idx[v.name]] = a
                trainable_variable_placeholders[name2idx[v.name]] = p
            else:
                extra_variables.append(v_float_value)
                extra_variable_assigns.append(a)
                extra_variable_assign_placeholders.append(p)

        extra_variable_assign = tf.group(*extra_variable_assigns)
        trainable_assign = tf.group(*trainable_assigns)
        update_op = tf.group(update_ops)

        return trainable_variables, trainable_variable_placeholders, trainable_assign, \
            extra_variables, extra_variable_assign_placeholders, \
            extra_variable_assign, update_op

    @staticmethod
    def _save_to_dir_for_unfreeze(folder, sess, graph,
                                  metric_tensors,
                                  batch_size_tensor,
                                  loss_tensor, inputs,
                                  trainable_variables,
                                  trainable_variable_placeholders,
                                  trainable_assign,
                                  extra_variables,
                                  extra_variable_assign_placeholders,
                                  extra_variable_assign,
                                  grads, update_op,
                                  additional_values):

        import tensorflow as tf
        from tensorflow import gfile
        saver = tf.train.Saver()
        if not os.path.isdir(folder):
            os.makedirs(folder)
        saver.save(sess, os.path.join(folder, "model"), write_meta_graph=False)

        meta = {
            "inputs": [i.name for i in inputs],
            "input_types": [i.dtype.as_datatype_enum for i in inputs],
            "metric_tensors": [t.name for t in metric_tensors],
            "batch_size_tensor": batch_size_tensor.name,
            "loss_tensor": loss_tensor.name,
            "variables": [v.name for v in trainable_variables],
            "variable_types": [v.dtype.as_datatype_enum for v in trainable_variable_placeholders],
            "variable_assign_placeholders": [v.name for v in trainable_variable_placeholders],
            "assign_variable_op": trainable_assign.name,
            "extra_variables": [v.name for v in extra_variables],
            "extra_variable_types": [v.dtype.as_datatype_enum for v
                                     in extra_variable_assign_placeholders],
            "extra_variable_assign_placeholders": [p.name for p in
                                                   extra_variable_assign_placeholders],
            "assign_extra_variable_op": extra_variable_assign.name,
            "grad_variables": [g.name for g in grads],
            "update_op": update_op.name,
            "restore_op": saver.saver_def.restore_op_name,
            "restore_path_placeholder": saver.saver_def.filename_tensor_name,
            "save_op": _to_operation_name(saver.saver_def.save_tensor_name),
            "save_path_placeholder": saver.saver_def.filename_tensor_name,
            "default_tensor_value": [_to_floats(v) for v in additional_values]
        }

        with open(os.path.join(folder, "training_meta.json"), "w") as f:
            f.write(json.dumps(meta))

        with gfile.GFile(os.path.join(folder, "model.meta"), "wb") as f:
            f.write(graph.as_graph_def().SerializeToString())

        return meta, saver

    @staticmethod
    def export_for_training(model_dir, loss_tensor, sess, inputs, grads, variables, graph,
                            tensors_with_value, metrics, updates):
        import tensorflow as tf

        inputs, additional_values = TFModel._expand_inputs(inputs, tensors_with_value, loss_tensor)
        metric_tensors, val_methods = TFModel._process_metrics(graph, metrics)
        grads = TFModel._process_grads(graph, grads)

        with graph.as_default():
            batch_size_tensor = tf.to_float(tf.shape(inputs[0])[0])

        trainable_variables, trainable_variable_placeholders, trainable_assign, \
            extra_variables, extra_variable_assign_placeholders, \
            extra_variable_assign, update_op = \
            TFModel._process_variables_for_unfreeze(graph, variables, updates)

        meta, saver = \
            TFModel._save_to_dir_for_unfreeze(model_dir, sess, graph,
                                              metric_tensors,
                                              batch_size_tensor,
                                              loss_tensor, inputs,
                                              trainable_variables,
                                              trainable_variable_placeholders,
                                              trainable_assign,
                                              extra_variables,
                                              extra_variable_assign_placeholders,
                                              extra_variable_assign,
                                              grads, update_op,
                                              additional_values)
        return meta, saver, val_methods

    @staticmethod
    def create_for_unfreeze(loss_tensor, sess, inputs, grads, variables, graph,
                            tensors_with_value, session_config, metrics, updates, model_dir):

        if model_dir is None:
            model_dir = tempfile.mkdtemp()
        else:
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

        meta, saver, val_methods = TFModel.export_for_training(model_dir, loss_tensor, sess,
                                                               inputs, grads, variables, graph,
                                                               tensors_with_value, metrics, updates)

        training_helper_layer = TFTrainingHelper(model_dir,
                                                 session_config, saver, meta, sess)

        criterion = IdentityCriterion()

        return TFModel(training_helper_layer, criterion, val_methods)


class TFOptimizer:
    def __init__(self, tf_model, optim_method,
                 sess=None, dataset=None,
                 val_split=0.0,
                 clip_norm=None, clip_value=None,
                 model_dir=None):
        """
        TFOptimizer is used for distributed training of TensorFlow
        on Spark/BigDL.

        Note that if grads and variables are not None, then they need to be sorted by name
        if you want to use multiple optimization methods for a TensorFlow model according to
        variable names.

        :param loss: The loss tensor of the TensorFlow model, should be a scalar
        :param optim_method: the optimization method to be used, such as bigdl.optim.optimizer.Adam
        :param sess: the current TensorFlow Session, if you want to used a pre-trained model, you
        should use the Session to load the pre-trained variables and pass it to TFOptimizer.
        """

        self.optim_method = optim_method
        self.sess = sess
        self.dataset = dataset

        self.clip_norm = clip_norm
        if clip_value is not None and not isinstance(clip_value, tuple):
            raise ValueError("The clip_value argument should be a tuple (min_value, max_value)")
        self.clip_constant = clip_value

        if self.dataset.batch_size <= 0:
            raise ValueError("You should set batch_size instead of batch_per_thread for training")

        self.model_dir = model_dir

        self.tf_model = tf_model

        batch_size = self.dataset.batch_size

        sample_rdd = self.dataset.get_training_data()

        if val_split != 0.0:
            training_rdd, val_rdd = sample_rdd.randomSplit([1 - val_split, val_split])
        else:
            training_rdd = sample_rdd
            val_rdd = self.dataset.get_validation_data()

        self.training_rdd = training_rdd
        self.val_rdd = val_rdd
        self.batch_size = batch_size

        self.estimator = Estimator(self.tf_model.training_helper_layer, self.optim_method,
                                   model_dir)

        if self.clip_norm:
            self.estimator.set_l2_norm_gradient_clipping(self.clip_norm)
        if self.clip_constant:
            min_value, max_value = self.clip_constant
            self.estimator.set_constant_gradient_clipping(min_value, max_value)

    @staticmethod
    def _get_arguments_from_loss(loss, optim_method, session, val_outputs, val_labels, val_method):
        import tensorflow as tf
        if session is None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        else:
            sess = session

        grads, variables = TFOptimizer._get_vars_grads(loss)
        all_required_inputs = _find_placeholders([loss])
        dataset = tf.get_collection(all_required_inputs[0].name)[0]

        inputs = nest.flatten(dataset._original_tensors)

        return [loss, optim_method, sess, dataset, inputs,
                grads, variables, loss.graph, val_outputs, val_labels, val_method]

    @staticmethod
    def _get_vars_grads(loss):
        import tensorflow as tf
        grads_vars = tf.train.GradientDescentOptimizer(0).compute_gradients(loss)
        grads_vars.sort(key=lambda grad_var: grad_var[1].name)
        variables = []
        grads = []
        for (grad, var) in grads_vars:
            if grad is not None:
                variables.append(var)
                grads.append(grad)
        return grads, variables

    @classmethod
    def from_loss(cls, loss, optim_method, session=None, val_outputs=None,
                  val_labels=None, val_method=None, val_split=0.0,
                  clip_norm=None, clip_value=None, metrics=None,
                  tensor_with_value=None, session_config=None, model_dir=None, updates=None):
        """
        Create a TFOptimizer from a TensorFlow loss tensor.
        The loss tensor must come from a TensorFlow graph that only takes TFDataset.tensors and
        the tensors in `tensor_with_value` as inputs.
        :param loss: The loss tensor of the TensorFlow model, should be a scalar
        :param optim_method: the optimization method to be used, such as bigdl.optim.optimizer.Adam
        :param session: the current TensorFlow Session, if you want to used a pre-trained model,
        you should use the Session to load the pre-trained variables and pass it to TFOptimizer.
        :param val_outputs: the validation output TensorFlow tensor to be used by val_methods
        :param val_labels: the validation label TensorFlow tensor to be used by val_methods
        :param val_method: the BigDL val_method(s) to be used.
        :param val_split: Float between 0 and 1. Fraction of the training data to be used as
        validation data.
        :param clip_norm: float >= 0. Gradients will be clipped when their L2 norm exceeds
        this value.
        :param clip_value: float >= 0. Gradients will be clipped when their absolute value
        exceeds this value.
        :param metrics: a dictionary. The key should be a string representing the metric's name
        and the value should be the corresponding TensorFlow tensor, which should be a scalar.
        :param tensor_with_value: a dictionary. The key is TensorFlow tensor, usually a
        placeholder, the value of the dictionary is a tuple of two elements. The first one of
        the tuple is the value to feed to the tensor in training phase and the second one
        is the value to feed to the tensor in validation phase.
        :return: a TFOptimizer
        """
        args = TFOptimizer._get_arguments_from_loss(loss, optim_method,
                                                    session, val_outputs,
                                                    val_labels, val_method)

        loss, optim_method, sess, dataset, inputs = args[:5]
        grads, variables, graph, val_outputs, val_labels, val_method = args[5:]
        if clip_value is not None:
            if isinstance(clip_value, float) or isinstance(clip_value, int):
                if clip_value <= 0:
                    ValueError("The clip_value argument should be positive number")
                clip_value = (-float(clip_value), float(clip_value))

            if not isinstance(clip_value, tuple):
                raise ValueError("The clip_value argument should be" +
                                 " a positive float/int which clips to" +
                                 " (-clip_value, clip_value); " +
                                 "or a tuple which clips to (min_value, max_value)")

        if val_method is not None:
            val_methods = to_list(val_method)
            if metrics is None:
                metrics = {}

            for i, method in enumerate(val_methods):
                metrics['bigdl_metirc_' + str(i)] = BigDLMetric(method, val_outputs, val_labels)

        tf_model = TFModel.create_for_unfreeze(loss, sess, inputs, grads, variables, graph,
                                               tensor_with_value, session_config, metrics,
                                               updates, model_dir)

        return cls(tf_model, optim_method, sess=sess, dataset=dataset, val_split=val_split,
                   clip_norm=clip_norm, clip_value=clip_value)

    @staticmethod
    def export_training_model(export_dir, loss, sess, inputs,
                              metrics=None, tensor_with_value=None, updates=None):

        grads, variables = TFOptimizer._get_vars_grads(loss)

        TFModel.export_for_training(export_dir, loss, sess, inputs, grads, variables, loss.graph,
                                    tensor_with_value, metrics, updates)
        logging.info("Exported TensorFlow model in {} for training".format(export_dir))

    @classmethod
    def from_keras(cls, keras_model, dataset, optim_method=None, val_split=0.0,
                   session_config=None, model_dir=None):
        """
        Create a TFOptimizer from a tensorflow.keras model. The model must be compiled.
        :param keras_model: the tensorflow.keras model, which must be compiled.
        :param dataset: a TFDataset
        :param optim_method: the optimization method to be used, such as bigdl.optim.optimizer.Adam
        :param val_split: Float between 0 and 1. Fraction of the training data to be used as
        validation data.
        :return:
        """
        import tensorflow.keras.backend as K

        if isinstance(dataset, MapDataset):
            raise ValueError("MapDataset is not supported for Keras Model for now, " +
                             "please warp the map_fn in a Keras layer in your keras model")

        model_inputs = keras_model.inputs
        if hasattr(keras_model, "targets"):
            model_targets = keras_model.targets
        else:
            model_targets = keras_model._targets

        inputs = model_inputs + model_targets

        loss = keras_model.total_loss
        variables = keras_model._collected_trainable_weights
        variables.sort(key=lambda variable: variable.name)
        keras_optimizer = keras_model.optimizer

        grads = K.gradients(loss, variables)
        if None in grads:
            raise ValueError('An operation has `None` for gradient. '
                             'Please make sure that all of your ops have a '
                             'gradient defined (i.e. are differentiable). '
                             'Common ops without gradient: '
                             'K.argmax, K.round, K.eval.')
        clip_norm = None
        clip_value = None
        if hasattr(keras_optimizer, 'clipnorm'):
            clip_norm = keras_optimizer.clipnorm
        if hasattr(keras_optimizer, 'clipvalue'):
            clip_value = (-keras_optimizer.clipvalue, keras_optimizer.clipvalue)

        sess = K.get_session()
        if optim_method is None:
            optim_method = keras_optimizer
        optim_method = to_bigdl_optim_method(optim_method)

        if keras_model.metrics and (dataset.get_validation_data() is not None or val_split != 0.0):
            if isinstance(keras_model.metrics, dict):
                raise ValueError(
                    "different metrics for different outputs are not supported right now")

            if dataset.get_validation_data() is None and val_split == 0.0:
                raise ValueError("Validation data is not specified. Please set " +
                                 "val_rdd in TFDataset, or set val_split larger than zero")

            if len(keras_model.outputs) > 1:
                if not all([name.endswith("loss") for name in keras_model.metrics_names]):
                    raise ValueError("metrics (except loss) for multi-head model is not supported")
                else:
                    bigdl_val_methods = [Loss()]
                    val_outputs = keras_model.outputs
                    val_labels = model_targets
            else:
                bigdl_val_methods = \
                    [to_bigdl_metric(m, keras_model.loss) for m in keras_model.metrics_names]
                val_outputs = keras_model.outputs
                val_labels = model_targets
        else:
            val_outputs = None
            val_labels = None
            bigdl_val_methods = None

        tensor_with_value = {
            K.learning_phase(): [True, False]
        }

        updates = keras_model.updates

        metrics = None

        if bigdl_val_methods is not None:
            val_methods = to_list(bigdl_val_methods)
            metrics = {}
            for i, method in enumerate(val_methods):
                metrics['bigdl_metirc_' + str(i)] = BigDLMetric(method, val_outputs, val_labels)

        tf_model = TFModel.create_for_unfreeze(loss, sess, inputs, grads, variables, loss.graph,
                                               tensor_with_value, session_config, metrics,
                                               updates, model_dir)

        return cls(tf_model, optim_method, sess=sess, dataset=dataset, val_split=val_split,
                   clip_norm=clip_norm, clip_value=clip_value)

    def set_constant_gradient_clipping(self, min_value, max_value):
        """
        Configure constant clipping settings.

        :param min_value: the minimum value to clip by
        :param max_value: the maxmimum value to clip by
        """
        self.estimator.set_constant_gradient_clipping(min_value, max_value)

    def set_gradient_clipping_by_l2_norm(self, clip_norm):
        """
        Configure L2 norm clipping settings.
        :param clip_norm: gradient L2-Norm threshold
        """
        self.estimator.set_l2_norm_gradient_clipping(clip_norm)

    def optimize(self, end_trigger=None, checkpoint_trigger=None):
        """
        Run the training loop of the this optimizer
        :param end_trigger: BigDL's Trigger to indicate when to stop the training.
        :param checkpoint_trigger: When to save a checkpoint and evaluate model.
        """
        if end_trigger is None:
            end_trigger = MaxEpoch(1)

        if checkpoint_trigger is None:
            checkpoint_trigger = EveryEpoch()

        if self.tf_model.val_methods is not None and self.val_rdd is not None:
            self.estimator.train_minibatch(train_set=self.training_rdd,
                                           criterion=self.tf_model.criterion,
                                           end_trigger=end_trigger,
                                           checkpoint_trigger=checkpoint_trigger,
                                           validation_set=self.val_rdd,
                                           validation_method=self.tf_model.val_methods)
        else:
            self.estimator.train_minibatch(train_set=self.training_rdd,
                                           criterion=self.tf_model.criterion,
                                           end_trigger=end_trigger)

        self.tf_model.training_helper_layer.get_weights_to_python()
