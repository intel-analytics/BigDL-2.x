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

from abc import ABC, abstractmethod


class Estimator(ABC):
    @abstractmethod
    def fit(self, data, epochs, **kwargs):
        pass

    @abstractmethod
    def predict(self, data, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, data, **kwargs):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def save(self, model_path):
        pass

    @abstractmethod
    def load(self, checkpoint, **kwargs):
        pass

    @abstractmethod
    def set_tensorboard(self, log_dir, app_name):
        """
        Set summary information during the training process for visualization purposes.
        Saved summary can be viewed via TensorBoard.
        In order to take effect, it needs to be called before fit.

        Training summary will be saved to 'log_dir/app_name/train'
        and validation summary (if any) will be saved to 'log_dir/app_name/validation'.

        # Arguments
        :param log_dir: The base directory path to store training and validation logs.
        :param app_name: The name of the application.
        """
        pass

    @abstractmethod
    def clear_gradient_clipping(self):
        """
        Clear gradient clipping parameters. In this case, gradient clipping will not be applied.
        In order to take effect, it needs to be called before fit.
        :return:
         """
        pass

    @abstractmethod
    def set_constant_gradient_clipping(self, min, max):
        """
        Set constant gradient clipping during the training process.
        In order to take effect, it needs to be called before fit.
        :param min: The minimum value to clip by.
        :param max: The maximum value to clip by.
        :return:
        """
        pass

    @abstractmethod
    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Clip gradient to a maximum L2-Norm during the training process.
        In order to take effect, it needs to be called before fit.
        :param clip_norm: Gradient L2-Norm threshold.
        :return:
        """
        pass

    @abstractmethod
    def get_train_summary(self, tag=None):
        """
        Get the scalar from model train summary
        Return list of summary data of [iteration_number, scalar_value, timestamp]
        # Arguments
        tag: The string variable represents the scalar wanted
        """
        pass

    @abstractmethod
    def get_validation_summary(self, tag=None):
        """
        Get the scalar from model validation summary
        Return list of summary data of [iteration_number, scalar_value, timestamp]

        Note: The metric and tag may not be consistent
        Please look up following form to pass tag parameter
        Left side is your metric during compile
        Right side is the tag you should pass
        'Accuracy'                  |   'Top1Accuracy'
        'BinaryAccuracy'            |   'Top1Accuracy'
        'CategoricalAccuracy'       |   'Top1Accuracy'
        'SparseCategoricalAccuracy' |   'Top1Accuracy'
        'AUC'                       |   'AucScore'
        'HitRatio'                  |   'HitRate@k' (k is Top-k)
        'Loss'                      |   'Loss'
        'MAE'                       |   'MAE'
        'NDCG'                      |   'NDCG'
        'TFValidationMethod'        |   '${name + " " + valMethod.toString()}'
        'Top5Accuracy'              |   'Top5Accuracy'
        'TreeNNAccuracy'            |   'TreeNNAccuracy()'
        'MeanAveragePrecision'      |   'MAP@k' (k is Top-k) (BigDL)
        'MeanAveragePrecision'      |   'PascalMeanAveragePrecision' (Zoo)
        'StatelessMetric'           |   '${name}'
        # Arguments
        tag: The string variable represents the scalar wanted
        """
        pass

    def save_tf_checkpoint(self, path):
        """
        Save tensorflow checkpoint in this estimator.
        :param path: tensorflow checkpoint path.
        """
        pass

    def save_keras_model(self, path, overwrite=True):
        """
        Save tensorflow keras model in this estimator.
        :param path: keras model save path.
        :param overwrite: Whether to silently overwrite any existing file at the target location.
        """
        pass

    @abstractmethod
    def load_orca_checkpoint(self, path, version):
        """
        Load specified Orca checkpoint.
        :param path: checkpoint directory which contains model.* and
        optimMethod-TFParkTraining.* files.
        :param version: checkpoint version, which is the suffix of model.* file,
        i.e., for modle.4 file, the version is 4.
        """
        pass

    @abstractmethod
    def load_latest_orca_checkpoint(self, path):
        """
        Load latest Orca checkpoint under specified directory.
        :param path: directory containing Orca checkpoint files.
        """
        pass

    @staticmethod
    def from_bigdl(*, model, loss=None, optimizer=None, feature_preprocessing=None,
                   label_preprocessing=None, model_dir=None):
        """
        Construct an Estimator with BigDL model, loss function and Preprocessing for feature and
        label data.
        :param model: BigDL Model to be trained.
        :param loss: BigDL criterion.
        :param optimizer: BigDL optimizer.
        :param feature_preprocessing: The param converts the data in feature column to a
               Tensor or to a Sample directly. It expects a List of Int as the size of the
               converted Tensor, or a Preprocessing[F, Tensor[T]]

               If a List of Int is set as feature_preprocessing, it can only handle the case that
               feature column contains the following data types:
               Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The
               feature data are converted to Tensors with the specified sizes before
               sending to the model. Internally, a SeqToTensor is generated according to the
               size, and used as the feature_preprocessing.

               Alternatively, user can set feature_preprocessing as Preprocessing[F, Tensor[T]]
               that transforms the feature data to a Tensor[T]. Some pre-defined Preprocessing are
               provided in package zoo.feature. Multiple Preprocessing can be combined as a
               ChainedPreprocessing.

               The feature_preprocessing will also be copied to the generated NNModel and applied
               to feature column during transform.
        :param label_preprocessing: similar to feature_preprocessing, but applies to Label data.
        :param model_dir: The path to save model. During the training, if checkpoint_trigger is
            defined and triggered, the model will be saved to model_dir.
        :return:
        """
        from zoo.orca.learn.bigdl.estimator import BigDLEstimatorWrapper
        return BigDLEstimatorWrapper(model=model, loss=loss, optimizer=optimizer,
                                     feature_preprocessing=feature_preprocessing,
                                     label_preprocessing=label_preprocessing, model_dir=model_dir)

    @staticmethod
    def from_torch(*,
                   model,
                   optimizer,
                   loss=None,
                   model_dir=None):
        from zoo.orca.learn.pytorch.estimator import PytorchSparkEstimatorWrapper
        return PytorchSparkEstimatorWrapper(model=model, loss=loss, optimizer=optimizer,
                                            model_dir=model_dir, bigdl_type="float")

    @staticmethod
    def from_openvino(*, model_path, batch_size=0):
        """
        Load an openVINO Estimator.

        :param model_path: String. The file path to the OpenVINO IR xml file.
        :param batch_size: Int. Set batch Size, default is 0 (use default batch size).
        """
        from zoo.orca.learn.openvino.estimator import OpenvinoEstimatorWrapper
        return OpenvinoEstimatorWrapper(model_path=model_path, batch_size=batch_size)

    @staticmethod
    def from_tf_graph(*, inputs, outputs=None,
                      labels=None, loss=None, optimizer=None,
                      clip_norm=None, clip_value=None,
                      metrics=None, updates=None,
                      sess=None, model_dir=None):
        """
        Create an Estimator for tesorflow graph.
        :param inputs: input tensorflow tensors.
        :param outputs: output tensorflow tensors.
        :param labels: label tensorflow tensors.
        :param loss: The loss tensor of the TensorFlow model, should be a scalar
        :param optimizer: tensorflow optimization method.
        :param clip_norm: float >= 0. Gradients will be clipped when their L2 norm exceeds
        this value.
        :param clip_value:  a float >= 0 or a tuple of two floats.
        If clip_value is a float, gradients will be clipped when their absolute value
        exceeds this value.
        If clip_value is a tuple of two floats, gradients will be clipped when their value less
        than clip_value[0] or larger than clip_value[1].
        :param metrics: metric tensor.
        :param sess: the current TensorFlow Session, if you want to used a pre-trained model,
        you should use the Session to load the pre-trained variables and pass it to estimator
        :param model_dir: location to save model checkpoint and summaries.
        :return: an Estimator object.
        """
        from zoo.orca.learn.tf.estimator import TFOptimizerWrapper
        return TFOptimizerWrapper(inputs=inputs,
                                  outputs=outputs,
                                  labels=labels,
                                  loss=loss,
                                  optimizer=optimizer,
                                  clip_norm=clip_norm,
                                  clip_value=clip_value,
                                  metrics=metrics, updates=updates,
                                  sess=sess,
                                  model_dir=model_dir
                                  )

    @staticmethod
    def from_keras(keras_model, metrics=None, model_dir=None, optimizer=None):
        """
        Create an Estimator from a tensorflow.keras model. The model must be compiled.
        :param keras_model: the tensorflow.keras model, which must be compiled.
        :param metrics: user specified metric.
        :param model_dir: location to save model checkpoint and summaries.
        :param optimizer: an optional bigdl optimMethod that will override the optimizer in
                          keras_model.compile
        :return: an Estimator object.
        """
        from zoo.orca.learn.tf.estimator import TFKerasWrapper
        return TFKerasWrapper(keras_model, metrics, model_dir, optimizer)
