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

from pyspark.ml.param.shared import *
from pyspark.ml.wrapper import JavaModel, JavaEstimator, JavaTransformer
from bigdl.optim.optimizer import SGD
from bigdl.util.common import *
from zoo.feature.common import *

if sys.version >= '3':
    long = int
    unicode = str


class HasBatchSize(Params):
    """
    Mixin for param batchSize: batch size.
    """

    # a placeholder to make it appear in the generated doc
    batchSize = Param(Params._dummy(), "batchSize", "batchSize (>= 0).")

    def __init__(self):
        super(HasBatchSize, self).__init__()
        #: param for batch size.
        self.batchSize = Param(self, "batchSize", "batchSize")
        self._setDefault(batchSize=1)

    def setBatchSize(self, val):
        """
        Sets the value of :py:attr:`batchSize`.
        """
        self._paramMap[self.batchSize] = val
        return self

    def getBatchSize(self):
        """
        Gets the value of batchSize or its default value.
        """
        return self.getOrDefault(self.batchSize)


class HasSamplePreprocessing:
    """
    Mixin for param samplePreprocessing
    """
    samplePreprocessing = None

    def __init__(self):
        super(HasSamplePreprocessing, self).__init__()

    def setSamplePreprocessing(self, val):
        """
        Sets samplePreprocessing
        """
        pythonBigDL_method_name = "setSamplePreprocessing"
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        self.samplePreprocessing = val
        return self

    def getSamplePreprocessing(self):
        return self.samplePreprocessing


class HasOptimMethod:

    optimMethod = SGD()

    def __init__(self):
        super(HasOptimMethod, self).__init__()

    def setOptimMethod(self, val):
        """
        Sets optimization method. E.g. SGD, Adam, LBFGS etc. from bigdl.optim.optimizer.
        default: SGD()
        """
        pythonBigDL_method_name = "setOptimMethod"
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        self.optimMethod = val
        return self

    def getOptimMethod(self):
        """
        Gets the optimization method
        """
        return self.optimMethod


class NNEstimator(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasBatchSize,
                  HasOptimMethod, HasSamplePreprocessing, JavaValue):
    """
    NNEstimator extends org.apache.spark.ml.Estimator and supports training a BigDL model with
    Spark DataFrame data. It can be integrated into a standard Spark ML Pipeline to enable
    users for combined usage with Spark MLlib.

    NNEstimator supports different feature and label data type through operation defined in
    Preprocessing. We provide pre-defined Preprocessing for popular data types like Array
    or Vector in package zoo.feature, while user can also develop customized Preprocess
    which extends from feature.common.Preprocessing. During fit, NNEstimator
    will extract feature and label data from input DataFrame and use the Preprocessing to prepare
    data for the model.
    Using the Preprocessing allows NNEstimator to cache only the raw data and decrease the
    memory consumption during feature conversion and training.

    More concrete examples are available in package com.intel.analytics.zoo.examples.nnframes
    """

    def __init__(self, model, criterion,
                 feature_preprocessing=SeqToTensor(),
                 label_preprocessing=SeqToTensor(),
                 jvalue=None, bigdl_type="float"):
        """
        Construct a NNEstimator with BigDL model, criterion and Preprocessing for feature and label
        data.
        :param model: BigDL Model to be trained.
        :param criterion: BigDL criterion.
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
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        super(NNEstimator, self).__init__()
        if type(feature_preprocessing) is list:
            assert(all(isinstance(x, int) for x in feature_preprocessing))
            feature_preprocessing = SeqToTensor(feature_preprocessing)

        if type(label_preprocessing) is list:
            assert(all(isinstance(x, int) for x in label_preprocessing))
            label_preprocessing = SeqToTensor(label_preprocessing)

        sample_preprocessing = FeatureLabelPreprocessing(feature_preprocessing, label_preprocessing)

        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), model, criterion, sample_preprocessing)
        self.model = model
        self.samplePreprocessing = sample_preprocessing
        self.bigdl_type = bigdl_type
        self._java_obj = self.value

        self.maxEpoch = Param(self, "maxEpoch", "number of max Epoch")
        self.learningRate = Param(self, "learningRate", "learning rate")
        self.learningRateDecay = Param(self, "learningRateDecay", "learning rate decay")
        self.cachingSample = Param(self, "cachingSample", "cachingSample")
        self._setDefault(maxEpoch=50, learningRate=1e-3, batchSize=1, learningRateDecay=0.0,
                         cachingSample=True)

        self.train_summary = None
        self.validation_config = None
        self.validation_summary = None

    def setSamplePreprocessing(self, val):
        """
        Sets the value of sample_preprocessing
        :param val: a Preprocesing[(Feature, Option(Label), Sample]
        """
        super(NNEstimator, self).setSamplePreprocessing(val)
        return self

    def setMaxEpoch(self, val):
        """
        Sets the value of :py:attr:`maxEpoch`.
        """
        self._paramMap[self.maxEpoch] = val
        return self

    def getMaxEpoch(self):
        """
        Gets the value of maxEpoch or its default value.
        """
        return self.getOrDefault(self.maxEpoch)

    def setLearningRate(self, val):
        """
        Sets the value of :py:attr:`learningRate`.
        """
        self._paramMap[self.learningRate] = val
        return self

    def getLearningRate(self):
        """
        Gets the value of learningRate or its default value.
        """
        return self.getOrDefault(self.learningRate)

    def setLearningRateDecay(self, val):
        """
        Sets the value of :py:attr:`learningRateDecay`.
        """
        self._paramMap[self.learningRateDecay] = val
        return self

    def getLearningRateDecay(self):
        """
        Gets the value of learningRateDecay or its default value.
        """
        return self.getOrDefault(self.learningRateDecay)

    def setCachingSample(self, val):
        """
        whether to cache the Samples after preprocessing. Default: True
        """
        self._paramMap[self.cachingSample] = val
        return self

    def isCachingSample(self):
        """
        Gets the value of cachingSample or its default value.
        """
        return self.getOrDefault(self.cachingSample)

    def setTrainSummary(self, val):
        """
        Statistics (LearningRate, Loss, Throughput, Parameters) collected during training for the
        training data, which can be used for visualization via Tensorboard.
        Use setTrainSummary to enable train logger. Then the log will be saved to
        logDir/appName/train as specified by the parameters of TrainSummary.
        Default: Not enabled

        :param summary: a TrainSummary object
        """
        pythonBigDL_method_name = "setTrainSummary"
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        self.train_summary = val
        return self

    def getTrainSummary(self):
        """
        Gets the train summary
        """
        return self.train_summary

    def setValidationSummary(self, val):
        """
        Statistics (LearningRate, Loss, Throughput, Parameters) collected during training for the
        validation data if validation data is set, which can be used for visualization via
        Tensorboard. Use setValidationSummary to enable validation logger. Then the log will be
        saved to logDir/appName/ as specified by the parameters of validationSummary.
        Default: None
        """
        pythonBigDL_method_name = "setValidationSummary"
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        self.validation_summary = val
        return self

    def getValidationSummary(self):
        """
        Gets the Validation summary
        """
        return self.validation_summary

    def setValidation(self, trigger, val_df, val_method, batch_size):
        """
        Set a validate evaluation during training

        :param trigger: validation interval
        :param val_df: validation dataset
        :param val_method: the ValidationMethod to use,e.g. "Top1Accuracy", "Top5Accuracy", "Loss"
        :param batch_size: validation batch size
        """
        pythonBigDL_method_name = "setValidation"
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value,
                      trigger, val_df, val_method, batch_size)
        self.validation_config = [trigger, val_df, val_method, batch_size]
        return self

    def getValidation(self):
        """
        Gets the validate configuration. If validation config has been set, getValidation will
        return a List of [ValidationTrigger, Validation data, Array[ValidationMethod[T]],
        batchsize]
        """
        return self.validation_config

    def clearGradientClipping(self):
        """
        Clear clipping params, in this case, clipping will not be applied.
        In order to take effect, it needs to be called before fit.
        """
        callBigDlFunc(self.bigdl_type, "nnEstimatorClearGradientClipping",
                      self.value)
        return self

    def setConstantGradientClipping(self, min, max):
        """
        Set constant gradient clipping during the training process.
        In order to take effect, it needs to be called before fit.

        # Arguments
        min: The minimum value to clip by. Float.
        max: The maximum value to clip by. Float.
        """
        callBigDlFunc(self.bigdl_type, "nnEstimatorSetConstantGradientClipping",
                      self.value,
                      float(min),
                      float(max))
        return self

    def setGradientClippingByL2Norm(self, clip_norm):
        """
        Clip gradient to a maximum L2-Norm during the training process.
        In order to take effect, it needs to be called before fit.

        # Arguments
        clip_norm: Gradient L2-Norm threshold. Float.
        """
        callBigDlFunc(self.bigdl_type, "nnEstimatorSetGradientClippingByL2Norm",
                      self.value,
                      float(clip_norm))
        return self

    def _create_model(self, java_model):
        # explicity reset SamplePreprocessing even though java_model already has the preprocessing,
        # so that python NNModel also has sample_preprocessing
        estPreprocessing = self.getSamplePreprocessing()
        nnModel = NNModel(model=self.model, feature_preprocessing=None, jvalue=java_model,
                          bigdl_type=self.bigdl_type) \
            .setSamplePreprocessing(ChainedPreprocessing([ToTuple(), estPreprocessing]))

        nnModel.setFeaturesCol(self.getFeaturesCol()) \
            .setPredictionCol(self.getPredictionCol()) \
            .setBatchSize(self.getBatchSize())
        return nnModel


class NNModel(JavaTransformer, HasFeaturesCol, HasPredictionCol, HasBatchSize,
              HasSamplePreprocessing, JavaValue):
    """
    NNModel extends Spark ML Transformer and supports BigDL model with Spark DataFrame.

    NNModel supports different feature data type through Preprocessing. Some common
    Preprocessing have been defined in com.intel.analytics.zoo.feature.

    After transform, the prediction column contains the output of the model as Array[T], where
    T (Double or Float) is decided by the model type.
    """
    def __init__(self, model, feature_preprocessing=SeqToTensor(), jvalue=None, bigdl_type="float"):
        """
        create a NNModel with a BigDL model
        :param model: trained BigDL model to use in prediction.
        :param feature_preprocessing: The param converts the data in feature column to a
                                      Tensor. It expects a List of Int as
                                      the size of the converted Tensor, or a
                                      Preprocessing[F, Tensor[T]]
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        super(NNModel, self).__init__()
        # initialize with Java NNModel
        if jvalue:
            assert feature_preprocessing is None
            self.value = jvalue
        # initialize with Python Model and preprocessing
        else:
            if type(feature_preprocessing) is list:
                assert(all(isinstance(x, int) for x in feature_preprocessing))
                feature_preprocessing = SeqToTensor(feature_preprocessing)

            sample_preprocessing = ChainedPreprocessing([feature_preprocessing, TensorToSample()])
            self.value = callBigDlFunc(
                bigdl_type, self.jvm_class_constructor(), model, sample_preprocessing)
            self.samplePreprocessing = sample_preprocessing

        self.model = model
        self._java_obj = self.value
        self.bigdl_type = bigdl_type

    def save(self, path):
        self._transfer_params_to_java()
        callBigDlFunc(self.bigdl_type, "saveNNModel", self.value, path)
        return self

    @staticmethod
    def load(path):
        jvalue = callBigDlFunc("float", "loadNNModel", path)
        return NNModel(model=None, feature_preprocessing=None, jvalue=jvalue)


class NNClassifier(NNEstimator):
    """
    NNClassifier is a specialized NNEstimator that simplifies the data format for
    classification tasks. It only supports label column of DoubleType, and the fitted
    NNClassifierModel will have the prediction column of DoubleType.
    """
    def __init__(self, model, criterion, feature_preprocessing=SeqToTensor(),
                 jvalue=None, bigdl_type="float"):
        """
        :param model: BigDL module to be optimized
        :param criterion: BigDL criterion method
        :param feature_preprocessing: The param converts the data in feature column to a
                                      Tensor. It expects a List of Int as
                                      the size of the converted Tensor, or a
                                      Preprocessing[F, Tensor[T]]
        :param bigdl_type(optional): Data type of BigDL model, "float"(default) or "double".
        """
        super(NNClassifier, self).__init__(
            model, criterion, feature_preprocessing, ScalarToTensor(), jvalue, bigdl_type)

    def setSamplePreprocessing(self, val):
        """
        Sets the value of sample_preprocessing
        :param val: a Preprocesing[(Feature, Option(Label), Sample]
        """
        super(NNClassifier, self).setSamplePreprocessing(val)
        return self

    def _create_model(self, java_model):
        # explicity reset SamplePreprocessing even though java_model already has the preprocessing,
        # so that python NNClassifierModel also has sample_preprocessing
        estPreprocessing = self.getSamplePreprocessing()
        classifierModel = NNClassifierModel(model=self.model, feature_preprocessing=None,
                                            jvalue=java_model, bigdl_type=self.bigdl_type) \
            .setSamplePreprocessing(ChainedPreprocessing([ToTuple(), estPreprocessing]))

        classifierModel.setFeaturesCol(self.getFeaturesCol()) \
            .setPredictionCol(self.getPredictionCol()) \
            .setBatchSize(self.getBatchSize())
        return classifierModel


class NNClassifierModel(NNModel):
    """
    NNClassifierModel is a specialized [[NNModel]] for classification tasks. The prediction
    column will have the datatype of Double.
    """
    def __init__(self,  model, feature_preprocessing=SeqToTensor(), jvalue=None,
                 bigdl_type="float"):
        """
        :param model: trained BigDL model to use in prediction.
        :param feature_preprocessing: The param converts the data in feature column to a
                                      Tensor. It expects a List of Int as
                                      the size of the converted Tensor, or a
                                      Preprocessing[F, Tensor[T]]
        :param jvalue: Java object create by Py4j
        :param bigdl_type(optional): Data type of BigDL model, "float"(default) or "double".
        """
        super(NNClassifierModel, self).__init__(model, feature_preprocessing, jvalue, bigdl_type)

    @staticmethod
    def load(path):
        jvalue = callBigDlFunc("float", "loadNNClassifierModel", path)
        return NNClassifierModel(model=None, feature_preprocessing=None, jvalue=jvalue)
