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

import pytest
from pyspark.ml import Pipeline

from zoo.pipeline.estimator import *

from bigdl.nn.layer import Sequential, View, Linear, LogSoftMax, SpatialConvolution
from bigdl.nn.criterion import ClassNLLCriterion
from bigdl.optim.optimizer import *
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.feature.common import FeatureSet
from zoo import init_nncontext, init_spark_conf
import zoo.common


class TestEstimator(ZooTestCase):
    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = init_spark_conf().setMaster("local[1]").setAppName("testEstimator")
        self.sc = init_nncontext(sparkConf)
        assert (self.sc.appName == "testEstimator")

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    @staticmethod
    def _create_cnn_model():
        model = Sequential()
        model.add(SpatialConvolution(3, 1, 5, 5))
        model.add(View([1 * 220 * 220]))
        model.add(Linear(1 * 220 * 220, 20))
        model.add(LogSoftMax())
        return model

    @staticmethod
    def _generate_image_data(data_num, img_shape):
        images = []
        labels = []
        for i in range(0, data_num):
            features = np.random.uniform(0, 1, img_shape)
            label = np.array([2])
            images.append(features)
            labels.append(label)
        return images, labels

    def test_estimator_train_imagefeature(self):
        batch_size = 8
        epoch_num = 5
        images, labels = TestEstimator._generate_image_data(data_num=8, img_shape=(200, 200, 3))

        image_frame = DistributedImageFrame(self.sc.parallelize(images),
                                            self.sc.parallelize(labels))

        transformer = Pipeline([BytesToMat(), Resize(256, 256), CenterCrop(224, 224),
                                ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                MatToTensor(), ImageFrameToSample(target_keys=['label'])])
        data_set = FeatureSet.image_frame(image_frame).transform(transformer)

        model = TestEstimator._create_cnn_model()

        optim_method = SGD(learningrate=0.01)

        estimator = Estimator(model, optim_method, "")
        estimator.set_constant_gradient_clipping(0.1, 1.2)
        estimator.train_imagefeature(train_set=data_set, criterion=ClassNLLCriterion(),
                                     end_trigger=MaxEpoch(epoch_num),
                                     checkpoint_trigger=EveryEpoch(),
                                     validation_set=data_set,
                                     validation_method=[Top1Accuracy()],
                                     batch_size=batch_size)
        predict_result = model.predict_image(image_frame.transform(transformer))
        assert (predict_result.get_predict().count(), 8)

    def test_estimator_train(self):
        batch_size = 8
        epoch_num = 5

        images, labels = TestEstimator._generate_image_data(data_num=8, img_shape=(3, 224, 224))

        image_rdd = self.sc.parallelize(images)
        labels = self.sc.parallelize(labels)

        sample_rdd = image_rdd.zip(labels).map(
            lambda img_label: zoo.common.Sample.from_ndarray(img_label[0], img_label[1]))

        data_set = FeatureSet.sample_rdd(sample_rdd)

        model = TestEstimator._create_cnn_model()

        optim_method = SGD(learningrate=0.01)

        estimator = Estimator(model, optim_method, "")
        estimator.set_constant_gradient_clipping(0.1, 1.2)
        estimator.train(train_set=data_set, criterion=ClassNLLCriterion(),
                        end_trigger=MaxEpoch(epoch_num),
                        checkpoint_trigger=EveryEpoch(),
                        validation_set=data_set,
                        validation_method=[Top1Accuracy()],
                        batch_size=batch_size)
        predict_result = model.predict(sample_rdd)
        assert (predict_result.count(), 8)


if __name__ == "__main__":
    pytest.main([__file__])
