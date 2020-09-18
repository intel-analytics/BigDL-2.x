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
from zoo.pipeline.nnframes import NNEstimator


class Estimator(object):
    def fit(self, data, epochs, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, **kwargs):
        pass

    def get_model(self):
        pass

    def save(self, checkpoint):
        pass

    def load(self, checkpoint):
        pass

    def clear_gradient_clipping(self):
        pass

    def set_constant_gradient_clipping(self, min, max):
        pass

    def set_l2_norm_gradient_clipping(self, clip_norm):
        pass

    def get_train_summary(self):
        pass

    def get_validation_summary(self):
        pass

    @staticmethod
    def from_bigdl(*, model, loss, optimizer=None, feature_preprocessing=None,
                   label_preprocessing=None, input_type="spark_dataframe"):
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
        :param input_type: The data type of training data. Only `spark_dataframe` and `
            spark_xshards` are supported
        :return:
        """

        if input_type == "spark_dataframe":
            return NNEstimatorWrapper(model=model, loss=loss, optimizer=optimizer,
                                      feature_preprocessing=feature_preprocessing,
                                      label_preprocessing=label_preprocessing)
        elif input_type == "spark_xshards":
            if optimizer is None:
                from bigdl.optim.optimizer import SGD
                optimizer = SGD()
            raise NotImplementedError
        else:
            raise ValueError("only spark_dataframe and spark_xshards input type are supported "
                             "for now")


class NNEstimatorWrapper(Estimator):
    def __init__(self, *, model, loss, optimizer=None, feature_preprocessing=None,
                 label_preprocessing=None):
        self.estimator = NNEstimator(model, loss, feature_preprocessing, label_preprocessing)
        if optimizer is not None:
            self.estimator.setOptimMethod(optimizer)
        self.model = None

    def fit(self, data, epochs, feature_cols="features", optimizer=None, batch_size=32,
            caching_sample=True, val_data=None, val_trigger=None, val_methods=None,
            train_summary_dir=None, val_summary_dir=None, app_name=None, checkpoint_path=None,
            checkpoint_trigger=None):
        from zoo.orca.learn.metrics import Metrics
        from zoo.orca.learn.trigger import Trigger
        if isinstance(feature_cols, list):
            if len(feature_cols) == 1:
                feature_cols = feature_cols[0]
            else:
                from pyspark.ml.feature import VectorAssembler
                assembler = VectorAssembler(
                    inputCols=feature_cols,
                    outputCol="features")
                data = assembler.transform(data)
                if val_data is not None:
                    val_data = assembler.transform(val_data)
                feature_cols = "features"

        self.estimator.setBatchSize(batch_size).setMaxEpoch(epochs)\
            .setCachingSample(caching_sample).setFeaturesCol(feature_cols)

        if optimizer is not None:
            self.estimator.setOptimMethod(optimizer)

        if val_data is not None:
            assert val_trigger is not None and val_methods is not None, \
                "You should provide val_trigger and val_methods if you provide val_data."
            val_trigger = Trigger.convert_trigger(val_trigger)
            val_methods = Metrics.convert_metrics_list(val_methods)
            self.estimator.setValidation(val_trigger, val_data, val_methods, batch_size)
        if train_summary_dir is not None:
            from bigdl.optim.optimizer import TrainSummary
            assert app_name is not None, \
                "You should provide app_name if you provide train_summary_dir"
            train_summary = TrainSummary(log_dir=train_summary_dir, app_name=app_name)
            self.estimator.setTrainSummary(train_summary)
        if val_summary_dir is not None:
            from bigdl.optim.optimizer import ValidationSummary
            assert app_name is not None, \
                "You should provide app_name if you provide val_summary_dir"
            val_summary = ValidationSummary(log_dir=val_summary_dir, app_name=app_name)
            self.estimator.setValidationSummary(val_summary)
        if checkpoint_path is not None:
            assert checkpoint_trigger is not None, \
                "You should provide checkpoint_trigger if you provide checkpoint_path"
            checkpoint_trigger = Trigger.convert_trigger(checkpoint_trigger)
            self.estimator.setCheckpoint(checkpoint_path, checkpoint_trigger)

        self.model = self.estimator.fit(data)

    def predict(self, data, batch_size=8, feature_cols="features", sample_preprocessing=None):
        if self.model is not None:
            if isinstance(feature_cols, list):
                if len(feature_cols) == 1:
                    feature_cols = feature_cols[0]
                else:
                    from pyspark.ml.feature import VectorAssembler
                    assembler = VectorAssembler(
                        inputCols=feature_cols,
                        outputCol="features")
                    data = assembler.transform(data)
                    feature_cols = "features"
            self.model.setBatchSize(batch_size).setFeaturesCol(feature_cols)
            if sample_preprocessing is not None:
                self.model.setSamplePreprocessing(sample_preprocessing)
            return self.model.transform(data)
        else:
            raise ValueError("You should fit before calling predict")

    def evaluate(self, data, **kwargs):
        pass

    def get_model(self):
        if self.model is not None:
            return self.model
        else:
            raise ValueError("You should fit before calling get_model")

    def save(self, checkpoint):
        if self.model is not None:
            self.model.model.saveModel(checkpoint + ".bigdl", checkpoint + ".bin", True)
        else:
            raise ValueError("You should fit before calling save")

    def load(self, checkpoint, optimizer=None, loss=None, feature_preprocessing=None,
             label_preprocessing=None):
        assert optimizer is not None and loss is not None, \
            "You should provide optimizer and loss function"
        from zoo.pipeline.api.net import Net
        from zoo.pipeline.nnframes import NNModel
        model = Net.load_bigdl(checkpoint + ".bigdl", checkpoint + ".bin")
        self.estimator = NNEstimator(model, loss, feature_preprocessing=feature_preprocessing,
                                     label_preprocessing=label_preprocessing)\
            .setOptimMethod(optimizer)
        self.model = NNModel(model, feature_preprocessing=feature_preprocessing)
        return self

    def clear_gradient_clipping(self):
        self.estimator.clearGradientClipping()

    def set_constant_gradient_clipping(self, min, max):
        self.estimator.setConstantGradientClipping(min, max)

    def set_l2_norm_gradient_clipping(self, clip_norm):
        self.estimator.setGradientClippingByL2Norm(clip_norm)

    def get_train_summary(self):
        return self.estimator.getTrainSummary()

    def get_validation_summary(self):
        return self.estimator.getValidationSummary()
