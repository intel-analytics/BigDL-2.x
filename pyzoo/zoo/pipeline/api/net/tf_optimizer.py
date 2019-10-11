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

from bigdl.nn.criterion import Criterion
from bigdl.nn.layer import Layer
from bigdl.util.common import to_list, JavaValue, callBigDlFunc
from bigdl.optim.optimizer import EveryEpoch, MaxEpoch, Optimizer
from zoo.pipeline.api.keras.engine.topology import to_bigdl_metric
from zoo.pipeline.api.net.utils import _find_placeholders, _check_the_same
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
        JavaValue.__init__(self, None, "float", metric_name, idx)


class BigDLMetric(object):

    def __init__(self, val_method, outputs, labels):
        self.val_method = val_method
        self.outputs = outputs
        self.labels = labels


class TFTrainingHelper(Layer):
    def __init__(self, path, configProto, assign, variable_placeholders, sess):
        if configProto is not None:
            byte_arr = bytearray(configProto.SerializeToString())
        else:
            byte_arr = None
        self.sess = sess
        self.assign = assign
        self.variable_placeholders = variable_placeholders
        super(TFTrainingHelper, self).__init__(None, "float", path, byte_arr)

    def get_weights_to_python(self):
        variables = self.get_weights()

        feed_dict = dict(zip(self.variable_placeholders, variables))
        self.sess.run(self.assign, feed_dict=feed_dict)


class TFTrainingHelper2(Layer):
    def __init__(self, path, config_proto, saver, meta, sess):
        self.saver = saver
        self.meta = meta
        self.export_dir = path
        self.sess = sess
        if config_proto is not None:
            byte_arr = bytearray(config_proto.SerializeToString())
        else:
            byte_arr = None
        super(TFTrainingHelper2, self).__init__(None, "float", path, byte_arr)

    def save_checkpoint(self):
        callBigDlFunc(self.bigdl_type, "saveCheckpoint",
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
            assert isinstance(session_config, tf.ConfigProto),\
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
    def _process_metrics(graph, metrics, loss, inputs):
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
            with graph.as_default():
                real_batch_size = tf.shape(inputs[0])[0]
            outputs.append(real_batch_size)

        with graph.as_default():
            outputs = [tf.to_float(output) for output in outputs]

        outputs.append(loss)
        return outputs, val_methods

    @staticmethod
    def _process_variables(graph, variables):
        import tensorflow as tf
        variable_placeholders = []
        with graph.as_default():
            assigns = []
            for v in variables:
                p = tf.placeholder(dtype=tf.float32, shape=v.shape)
                a = tf.assign(v, p)
                variable_placeholders.append(p)
                assigns.append(a)
            assign = tf.group(*assigns)
        return assign, variable_placeholders

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
                                  outputs, inputs,
                                  trainable_variables,
                                  trainable_variable_placeholders,
                                  trainable_assign,
                                  extra_variables,
                                  extra_variable_assign_placeholders,
                                  extra_variable_assign,
                                  grads, update_op, additional_values):

        import tensorflow as tf
        from tensorflow import gfile
        saver = tf.train.Saver()
        if not os.path.isdir(folder):
            os.makedirs(folder)
        saver.save(sess, os.path.join(folder, "model"), write_meta_graph=False)

        output_names = [o.name for o in outputs]
        input_names = [i.name for i in inputs]

        meta = {
            "inputs": input_names,
            "input_types": [i.dtype.as_datatype_enum for i in inputs],
            "outputs": output_names,
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
    def create(loss, sess, inputs, grads, variables, graph,
               tensors_with_value, session_config, metrics, updates):

        import tensorflow as tf
        from zoo.util.tf import export_tf

        inputs, additional_values = TFModel._expand_inputs(inputs, tensors_with_value, loss)
        session_config = TFModel._process_session_config(session_config)
        grads = TFModel._process_grads(graph, grads)

        outputs, val_methods = TFModel._process_metrics(graph, metrics, loss, inputs)

        assign, variable_placeholders = TFModel._process_variables(graph, variables)

        export_dir = tempfile.mkdtemp()
        export_tf(sess, export_dir,
                  inputs=inputs,
                  outputs=grads + outputs)

        variable_names = [v.name for v in variables]
        grad_names = [g.name for g in grads]
        output_names = [o.name for o in outputs]

        def to_floats(vs):
            return [float(v) for v in vs]

        meta = {
            "input_names": [i.name for i in inputs],
            "output_names": output_names,
            "variables": variable_names,
            "grad_variables": grad_names,
            "default_tensor_values": [to_floats(v) for v in additional_values]
        }

        with open(os.path.join(export_dir, "training_meta.json"), "w") as f:
            f.write(json.dumps(meta))

        training_helper_layer = TFTrainingHelper(export_dir,
                                                 session_config,
                                                 assign,
                                                 variable_placeholders, sess)

        criterion = IdentityCriterion()

        return TFModel(training_helper_layer, criterion, val_methods)

    @staticmethod
    def create_for_unfreeze(loss, sess, inputs, grads, variables, graph,
                            tensors_with_value, session_config, metrics, updates):

        inputs, additional_values = TFModel._expand_inputs(inputs, tensors_with_value, loss)
        session_config = TFModel._process_session_config(session_config)
        grads = TFModel._process_grads(graph, grads)

        outputs, val_methods = TFModel._process_metrics(graph, metrics, loss, inputs)

        trainable_variables, trainable_variable_placeholders, trainable_assign, \
            extra_variables, extra_variable_assign_placeholders, \
            extra_variable_assign, update_op = \
            TFModel._process_variables_for_unfreeze(graph, variables, updates)

        folder = tempfile.mkdtemp()
        meta, saver = \
            TFModel._save_to_dir_for_unfreeze(folder, sess, graph,
                                              outputs, inputs,
                                              trainable_variables,
                                              trainable_variable_placeholders,
                                              trainable_assign,
                                              extra_variables,
                                              extra_variable_assign_placeholders,
                                              extra_variable_assign,
                                              grads, update_op, additional_values)

        training_helper_layer = TFTrainingHelper2(folder,
                                                  session_config, saver, meta, sess)

        criterion = IdentityCriterion()

        return TFModel(training_helper_layer, criterion, val_methods)


class TFOptimizer:
    def __init__(self, loss, optim_method, sess=None, dataset=None, inputs=None,
                 grads=None, variables=None, graph=None,
                 val_outputs=None, val_labels=None, val_method=None, val_split=0.0,
                 tensors_with_value=None, session_config=None,
                 clip_norm=None, clip_value=None, metrics=None, updates=None, freeze=False):
        """
        TFOptimizer is used for distributed training of TensorFlow
        on Spark/BigDL.

        :param loss: The loss tensor of the TensorFlow model, should be a scalar
        :param optim_method: the optimization method to be used, such as bigdl.optim.optimizer.Adam
        :param sess: the current TensorFlow Session, if you want to used a pre-trained model, you
        should use the Session to load the pre-trained variables and pass it to TFOptimizer.
        """

        if dataset is None:
            args = TFOptimizer._get_arguments_from_loss(loss, optim_method, sess,
                                                        val_outputs, val_labels, val_method)
            loss, optim_method, sess, dataset, inputs = args[:5]
            grads, variables, graph, val_outputs, val_labels, val_method = args[5:]

        self.optim_method = optim_method
        self.sess = sess
        self.dataset = dataset
        self.graph = graph

        self.clip_norm = clip_norm
        if clip_value is not None and not isinstance(clip_value, tuple):
            raise ValueError("The clip_value argument should be a tuple (min_value, max_value)")
        self.clip_constant = clip_value

        if self.dataset.batch_size <= 0:
            raise ValueError("You should set batch_size instead of batch_per_thread for training")

        if val_method is not None:
            val_methods = to_list(val_method)
            if metrics is None:
                metrics = {}

            for i, method in enumerate(val_methods):
                metrics['bigdl_metirc_' + str(i)] = BigDLMetric(method, val_outputs, val_labels)

        if freeze:
            self.tf_model = TFModel.create(loss,
                                           sess, inputs, grads, variables, graph,
                                           tensors_with_value, session_config,
                                           metrics, updates)
        else:
            self.tf_model = TFModel.create_for_unfreeze(loss, sess, inputs, grads,
                                                        variables, graph, tensors_with_value,
                                                        session_config, metrics, updates)

        batch_size = self.dataset.batch_size

        sample_rdd = self.dataset.get_training_data()

        if val_split != 0.0:
            training_rdd, val_rdd = sample_rdd.randomSplit([1 - val_split, val_split])
        else:
            training_rdd = sample_rdd
            val_rdd = self.dataset.get_validation_data()

        if self.tf_model.val_methods is not None and val_rdd is not None:

            self.optimizer = Optimizer.create(self.tf_model.training_helper_layer,
                                              training_rdd,
                                              IdentityCriterion(),
                                              batch_size=batch_size,
                                              optim_method=self.optim_method)
            self.optimizer.set_validation(self.dataset.batch_size,
                                          val_rdd,
                                          EveryEpoch(),
                                          self.tf_model.val_methods)
        else:
            self.optimizer = Optimizer.create(self.tf_model.training_helper_layer,
                                              training_rdd,
                                              IdentityCriterion(),
                                              batch_size=batch_size,
                                              optim_method=self.optim_method)

        if self.clip_norm:
            self.optimizer.set_gradclip_l2norm(self.clip_norm)
        if self.clip_constant:
            min_value, max_value = self.clip_constant
            self.optimizer.set_gradclip_const(min_value, max_value)

    @staticmethod
    def _get_arguments_from_loss(loss, optim_method, session, val_outputs, val_labels, val_method):
        import tensorflow as tf
        if session is None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        else:
            sess = session
        grads_vars = tf.train.GradientDescentOptimizer(0).compute_gradients(loss)
        variables = []
        grads = []
        for (grad, var) in grads_vars:
            if grad is not None:
                variables.append(var)
                grads.append(grad)

        all_required_inputs = _find_placeholders([loss])
        dataset = tf.get_collection(all_required_inputs[0].name)[0]

        inputs = nest.flatten(dataset._original_tensors)

        return [loss, optim_method, sess, dataset, inputs,
                grads, variables, loss.graph, val_outputs, val_labels, val_method]

    @classmethod
    def from_loss(cls, loss, optim_method, session=None, val_outputs=None,
                  val_labels=None, val_method=None, val_split=0.0,
                  clip_norm=None, clip_value=None, metrics=None,
                  tensor_with_value=None, **kwargs):
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

        return cls(*(args + [val_split]),
                   tensors_with_value=tensor_with_value,
                   clip_norm=clip_norm,
                   clip_value=clip_value, metrics=metrics, **kwargs)

    @classmethod
    def from_keras(cls, keras_model, dataset, optim_method=None, val_spilt=0.0, **kwargs):
        """
        Create a TFOptimizer from a tensorflow.keras model. The model must be compiled.
        :param keras_model: the tensorflow.keras model, which must be compiled.
        :param dataset: a TFDataset
        :param optim_method: the optimization method to be used, such as bigdl.optim.optimizer.Adam
        :param val_spilt: Float between 0 and 1. Fraction of the training data to be used as
        validation data.
        :return:
        """
        import tensorflow.keras.backend as K
        loss = keras_model.total_loss
        inputs = keras_model.inputs + keras_model.targets

        variables = keras_model._collected_trainable_weights
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
        optim_method = TFOptimizer.to_bigdl_optim_method(optim_method)

        if keras_model.metrics and (dataset.get_validation_data() is not None or val_spilt != 0.0):
            if isinstance(keras_model.metrics, dict):
                raise ValueError(
                    "different metrics for different outputs are not supported right now")

            if dataset.get_validation_data() is None and val_spilt == 0.0:
                raise ValueError("Validation data is not specified. Please set " +
                                 "val_rdd in TFDataset, or set val_split larger than zero")
            bigdl_val_methods = \
                [to_bigdl_metric(m, keras_model.loss) for m in keras_model.metrics_names]
            val_outputs = keras_model.outputs
            val_labels = keras_model.targets
        else:
            val_outputs = None
            val_labels = None
            bigdl_val_methods = None

        tensor_with_value = {
            K.learning_phase(): [True, False]
        }

        updates = keras_model.updates

        return cls(loss, optim_method, sess, dataset, inputs,
                   grads, variables, loss.graph, val_outputs, val_labels,
                   bigdl_val_methods, val_spilt,
                   tensors_with_value=tensor_with_value,
                   clip_norm=clip_norm,
                   clip_value=clip_value, updates=updates, **kwargs)

    @staticmethod
    def to_bigdl_optim_method(koptim_method):
        # koptim_method is always an object
        import tensorflow.keras.backend as K
        import tensorflow.keras.optimizers as koptimizers
        import bigdl.optim.optimizer as boptimizer
        import tensorflow.train as tftrain
        import tensorflow as tf
        from tensorflow.python.keras.optimizers import TFOptimizer

        if isinstance(koptim_method, TFOptimizer):
            koptim_method = koptim_method.optimizer

        if isinstance(koptim_method, boptimizer.OptimMethod):
            return koptim_method
        elif isinstance(koptim_method, koptimizers.Optimizer):
            lr = float(K.eval(koptim_method.lr))
            decay = float(K.eval(koptim_method.decay))
            if isinstance(koptim_method, koptimizers.Adagrad):
                warnings.warn("For Adagrad, we don't support epsilon for now")
                return boptimizer.Adagrad(learningrate=lr,
                                          learningrate_decay=decay)
            elif isinstance(koptim_method, koptimizers.SGD):
                momentum = float(K.eval(koptim_method.momentum))
                return boptimizer.SGD(learningrate=lr,
                                      learningrate_decay=decay,
                                      momentum=momentum,
                                      nesterov=koptim_method.nesterov)
            elif isinstance(koptim_method, koptimizers.Adam):
                beta1 = float(K.eval(koptim_method.beta_1))
                beta2 = float(K.eval(koptim_method.beta_2))
                return boptimizer.Adam(learningrate=lr,
                                       learningrate_decay=decay,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=koptim_method.epsilon)
            elif isinstance(koptim_method, koptimizers.RMSprop):
                rho = float(K.eval(koptim_method.rho))
                return boptimizer.RMSprop(learningrate=lr,
                                          learningrate_decay=decay,
                                          decayrate=rho,
                                          epsilon=koptim_method.epsilon)
            elif isinstance(koptim_method, koptimizers.Adadelta):
                warnings.warn(
                    "For Adadelta, we don't support learning rate and learning rate decay for now")
                return boptimizer.Adadelta(decayrate=koptim_method.rho,
                                           epsilon=koptim_method.epsilon)
            elif isinstance(koptim_method, koptimizers.Adamax):
                beta1 = float(K.eval(koptim_method.beta_1))
                beta2 = float(K.eval(koptim_method.beta_2))
                warnings.warn("For Adamax, we don't support learning rate decay for now")
                return boptimizer.Adamax(learningrate=lr,
                                         beta1=beta1,
                                         beta2=beta2,
                                         epsilon=koptim_method.epsilon)
        elif isinstance(koptim_method, tftrain.Optimizer):
            def get_value(v):
                if isinstance(v, (tf.Tensor, tf.SparseTensor, tf.Variable)):
                    return float(K.eval(v))
                else:
                    return float(v)

            if isinstance(koptim_method, tftrain.GradientDescentOptimizer):
                lr = get_value(koptim_method._learning_rate)
                return boptimizer.SGD(learningrate=lr)
            elif isinstance(koptim_method, tftrain.MomentumOptimizer):
                lr = get_value(koptim_method._learning_rate)
                momentum = get_value(koptim_method._momentum)
                use_nesterov = koptim_method._use_nesterov
                return boptimizer.SGD(learningrate=lr, momentum=momentum, nesterov=use_nesterov)
            elif isinstance(koptim_method, tftrain.AdagradOptimizer):
                lr = get_value(koptim_method._learning_rate)
                return boptimizer.Adagrad(learningrate=lr)
            elif isinstance(koptim_method, tftrain.AdamOptimizer):
                lr = get_value(koptim_method._lr)
                beta1 = get_value(koptim_method._beta1)
                beta2 = get_value(koptim_method._beta2)
                epsilon = get_value(koptim_method._epsilon)
                return boptimizer.Adam(learningrate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
            elif isinstance(koptim_method, tftrain.RMSPropOptimizer):
                lr = get_value(koptim_method._learning_rate)
                decay = get_value(koptim_method._decay)
                momentum = get_value(koptim_method._momentum)
                epsilon = get_value(koptim_method._epsilon)
                centered = get_value(koptim_method._centered)
                if momentum != 0.0 or centered:
                    warnings.warn(
                        "For RMSPropOptimizer, we don't support momentum and centered for now")
                return boptimizer.RMSprop(learningrate=lr,
                                          learningrate_decay=decay,
                                          epsilon=epsilon)
            elif isinstance(koptim_method, tftrain.AdadeltaOptimizer):
                lr = get_value(koptim_method._lr)
                rho = get_value(koptim_method._rho)
                epsilon = get_value(koptim_method._epsilon)
                warnings.warn(
                    "For Adadelta, we don't support learning rate for now")
                return boptimizer.Adadelta(decayrate=rho, epsilon=epsilon)

        raise ValueError("We don't support %s for now" % koptim_method)

    def set_train_summary(self, summary):
        """
        Set train summary. A TrainSummary object contains information
        necessary for the optimizer to know how often the logs are recorded,
        where to store the logs and how to retrieve them, etc. For details,
        refer to the docs of TrainSummary.
        :param summary: a TrainSummary object
        """
        self.optimizer.set_train_summary(summary)

    def set_val_summary(self, summary):
        """
        Set validation summary. A ValidationSummary object contains information
        necessary for the optimizer to know how often the logs are recorded,
        where to store the logs and how to retrieve them, etc. For details,
        refer to the docs of ValidationSummary.

        :param summary: a ValidationSummary object
        """
        self.optimizer.set_val_summary(summary)

    def set_constant_gradient_clipping(self, min_value, max_value):
        """
        Configure constant clipping settings.

        :param min_value: the minimum value to clip by
        :param max_value: the maxmimum value to clip by
        """
        self.optimizer.set_gradclip_const(min_value, max_value)

    def set_gradient_clipping_by_l2_norm(self, clip_norm):
        """
        Configure L2 norm clipping settings.
        :param clip_norm: gradient L2-Norm threshold
        """
        self.optimizer.set_gradclip_l2norm(clip_norm)

    def optimize(self, end_trigger=None):
        """
        Run the training loop of the this optimizer
        :param end_trigger: BigDL's Trigger to indicate when to stop the training.
        """
        if end_trigger is None:
            end_trigger = MaxEpoch(1)

        self.optimizer.set_end_when(end_trigger)

        self.optimizer.optimize()

        self.tf_model.training_helper_layer.get_weights_to_python()
