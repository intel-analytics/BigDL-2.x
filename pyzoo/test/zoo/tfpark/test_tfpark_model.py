import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
import tensorflow as tf
import numpy as np

from zoo.pipeline.api.net import TFDataset
from zoo.tfpark.model import Model


class TestTFPark(ZooTestCase):

    def create_model(self):
        data = tf.keras.layers.Input(shape=[10])

        x = tf.keras.layers.Flatten()(data)
        x = tf.keras.layers.Dense(10, activation='relu')(x)
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=data, outputs=predictions)
        model.compile(optimizer='rmsprop',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def create_training_data(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))
        return x, y

    def create_training_dataset(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))

        rdd_x = self.sc.parallelize(x)
        rdd_y = self.sc.parallelize(y)

        rdd = rdd_x.zip(rdd_y).map(lambda x: list(x))

        dataset = TFDataset.from_rdd(rdd,
                                     names=["features", "labels"],
                                     shapes=[[10], []],
                                     types=[tf.float32, tf.int32],
                                     batch_size=4,
                                     val_rdd=rdd
                                     )
        return dataset

    def create_evaluation_dataset(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))

        rdd_x = self.sc.parallelize(x)
        rdd_y = self.sc.parallelize(y)

        rdd = rdd_x.zip(rdd_y).map(lambda x: list(x))

        dataset = TFDataset.from_rdd(rdd,
                                     names=["features", "labels"],
                                     shapes=[[10], []],
                                     types=[tf.float32, tf.int32],
                                     batch_per_thread=1
                                     )
        return dataset

    def create_predict_dataset(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)

        rdd = self.sc.parallelize(x)

        rdd = rdd.map(lambda x: [x])

        dataset = TFDataset.from_rdd(rdd,
                                     names=["features"],
                                     shapes=[[10]],
                                     types=[tf.float32],
                                     batch_per_thread=1
                                     )
        return dataset

    def test_training_with_ndarray(self):

        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        x, y = self.create_training_data()

        model.fit(x, y, batch_size=2)

    def test_training_with_ndarry_distributed(self):
        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        x, y = self.create_training_data()

        model.fit(x, y, batch_size=4, distributed=True)

    def test_training_with_validation_data(self):

        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        x, y = self.create_training_data()

        val_x, val_y = self.create_training_data()

        model.fit(x, y, validation_data=(val_x, val_y), batch_size=4)

    def test_training_with_validation_data_distributed(self):

        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        x, y = self.create_training_data()

        val_x, val_y = self.create_training_data()

        model.fit(x, y, validation_data=(val_x, val_y), batch_size=4, distributed=True)

    def test_training_and_validation_with_dataset(self):
        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        dataset = self.create_training_dataset()

        model.fit(dataset)

    def test_evaluate_with_ndarray(self):

        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        model.fit(x, y, batch_size=4, epochs=10)

        results_after = model.evaluate(x, y)

        assert results_pre[0] > results_after[0]

    def test_evaluate_with_ndarray_distributed(self):

        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        model.fit(x, y, batch_size=4, epochs=10)

        results_after = model.evaluate(x, y, distributed=True)

        assert results_pre[0] > results_after[0]

    def test_evaluate_and_distributed_evaluate(self):

        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        results_after = model.evaluate(x, y, distributed=True)

        assert np.square(results_pre[0] - results_after[0]) < 0.000001
        assert np.square(results_pre[1] - results_after[1]) < 0.000001

    def test_evaluate_with_dataset(self):

        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        dataset = self.create_evaluation_dataset()

        results_after = model.evaluate(dataset)

        assert np.square(results_pre[0] - results_after[0]) < 0.000001
        assert np.square(results_pre[1] - results_after[1]) < 0.000001

    def test_predict_with_ndarray(self):

        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        pred_y = np.argmax(model.predict(x), axis=1)

        acc = np.average((pred_y == y))

        assert np.square(acc - results_pre[1]) < 0.000001

    def test_predict_with_ndarray_distributed(self):

        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        pred_y = np.argmax(model.predict(x, distributed=True), axis=1)

        acc = np.average((pred_y == y))

        assert np.square(acc - results_pre[1]) < 0.000001

    def test_predict_with_dataset(self):

        keras_model = self.create_model()
        model = Model.from_keras(keras_model)

        x, y = self.create_training_data()

        results_pre = model.evaluate(x, y)

        pred_y = np.argmax(np.array(model.predict(self.create_predict_dataset()).collect()), axis=1)

        acc = np.average((pred_y == y))

        assert np.square(acc - results_pre[1]) < 0.000001


if __name__ == "__main__":
    pytest.main([__file__])
