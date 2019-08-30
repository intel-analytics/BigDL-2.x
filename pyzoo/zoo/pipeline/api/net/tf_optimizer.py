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
from py4j.protocol import Py4JJavaError

from bigdl.nn.criterion import Criterion
from bigdl.nn.layer import Layer
from bigdl.util.common import to_list, JavaValue
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
    def __init__(self, path, configProto, assign, variable_placeholders):
        if configProto is not None:
            byte_arr = bytearray(configProto.SerializeToString())
        else:
            byte_arr = None
        self.assign = assign
        self.variable_placeholders = variable_placeholders
        super(TFTrainingHelper, self).__init__(None, "float", path, byte_arr)


class TFModel(object):

    def __init__(self, training_helper_layer, criterion, val_methods):

        self.training_helper_layer = training_helper_layer
        self.criterion = criterion
        self.val_methods = val_methods

    @staticmethod
    def create(loss, sess, inputs, grads, variables, graph,
               tensors_with_value, session_config, metrics):

        import tensorflow as tf
        from zoo.util.tf import export_tf
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

        if session_config is not None:
            import tensorflow as tf
            assert isinstance(session_config, tf.ConfigProto),\
                "session_config should be a tf.ConfigProto"
            session_config.use_per_session_threads = True
        session_config = session_config

        from zoo.util.tf import process_grad
        grads = [process_grad(grad) for grad in grads]

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

        outputs.append(loss)

        export_dir = tempfile.mkdtemp()
        # export_dir = "/tmp/training_helper"
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
        variable_placeholders = []
        with graph.as_default():
            assigns = []
            for v in variables:
                p = tf.placeholder(dtype=tf.float32, shape=v.shape)
                a = tf.assign(v, p)
                variable_placeholders.append(p)
                assigns.append(a)
            assign = tf.group(*assigns)
        assign = assign
        training_helper_layer = TFTrainingHelper(export_dir, session_config, assign, variable_placeholders)

        criterion = IdentityCriterion()

        return TFModel(training_helper_layer, criterion, val_methods)


class TFOptimizer:
    def __init__(self, loss, optim_method, sess=None, dataset=None, inputs=None,
                 grads=None, variables=None, graph=None,
                 val_outputs=None, val_labels=None, val_method=None, val_split=0.0,
                 tensors_with_value=None, session_config=None,
                 clip_norm=None, clip_value=None, metrics=None):
        '''
        TFOptimizer is used for distributed training of TensorFlow
        on Spark/BigDL.

        :param loss: The loss tensor of the TensorFlow model, should be a scalar
        :param optim_method: the optimization method to be used, such as bigdl.optim.optimizer.Adam
        :param sess: the current TensorFlow Session, if you want to used a pre-trained model, you
        should use the Session to load the pre-trained variables and pass it to TFOptimizer.
        '''

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
            if metrics is not None:
                metrics['bigdl_metric'] = BigDLMetric(val_method, val_outputs, val_labels)
            else:
                metrics = {'bigdl_metric': BigDLMetric(val_method, val_outputs, val_labels)}

        self.tf_model = TFModel.create(loss,
                                       sess, inputs, grads, variables, graph,
                                       tensors_with_value, session_config,
                                       metrics)

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

        _check_the_same(all_required_inputs, inputs)

        return [loss, optim_method, sess, dataset, inputs,
                grads, variables, loss.graph, val_outputs, val_labels, val_method]

    @classmethod
    def from_loss(cls, loss, optim_method, session=None, val_outputs=None,
                  val_labels=None, val_method=None, val_split=0.0,
                  clip_norm=None, clip_value=None, metrics=None, **kwargs):
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
                   clip_norm=clip_norm,
                   clip_value=clip_value, metrics=metrics, **kwargs)

    @classmethod
    def from_keras(cls, keras_model, dataset, optim_method=None, val_spilt=0.0, **kwargs):
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

        return cls(loss, optim_method, sess, dataset, inputs,
                   grads, variables, loss.graph, val_outputs, val_labels,
                   bigdl_val_methods, val_spilt,
                   tensors_with_value=tensor_with_value,
                   clip_norm=clip_norm,
                   clip_value=clip_value, **kwargs)

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
        self.optimizer.set_train_summary(summary)

    def set_val_summary(self, summary):
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
        if end_trigger is None:
            end_trigger = MaxEpoch(1)

        self.optimizer.set_end_when(end_trigger)

        self.optimizer.optimize()

        variables = self.tf_model.training_helper_layer.get_weights()

        feed_dict = dict(zip(self.tf_model.training_helper_layer.variable_placeholders, variables))
        self.sess.run(self.tf_model.training_helper_layer.assign, feed_dict=feed_dict)
