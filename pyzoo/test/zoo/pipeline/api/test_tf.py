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


from bigdl.optim.optimizer import Adam, MaxEpoch
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.pipeline.api.net import Net, TFNet, TFDataset, TFOptimizer
from bigdl.util.common import *

np.random.seed(1337)  # for reproducibility


class TestTF(ZooTestCase):

    def test_init_tf_net(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
        tfnet_path = os.path.join(resource_path, "tfnet")
        net = TFNet.from_export_folder(tfnet_path)
        output = net.forward(np.random.rand(2, 4))
        assert output.shape == (2, 2)

    def test_from_folder_load_tf(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
        tfnet_path = os.path.join(resource_path, "tfnet")
        net = Net.load_tf(tfnet_path)
        output = net.forward(np.random.rand(2, 4))
        assert output.shape == (2, 2)

    def test_for_scalar(self):
        import tensorflow as tf
        with tf.Graph().as_default():
            input1 = tf.placeholder(dtype=tf.float32, shape=())
            output = input1 + 1
            sess = tf.Session()
            net = TFNet.from_session(sess, [input1], [output])
            sess.close()
        out_value = net.forward(np.array(1.0))
        assert len(out_value.shape) == 0

        # the following test would fail on bigdl 0.6.0 due to a bug in bigdl,
        # comment it out for now

        # out_value = net.predict(np.array([1.0])).first()
        # assert len(out_value.shape) == 0

    def test_init_tfnet_from_session(self):
        import tensorflow as tf
        with tf.Graph().as_default():
            input1 = tf.placeholder(dtype=tf.float32, shape=(None, 2))
            label1 = tf.placeholder(dtype=tf.float32, shape=(None, 1))
            hidden = tf.layers.dense(input1, 4)
            output = tf.layers.dense(hidden, 1)
            loss = tf.reduce_mean(tf.square(output - label1))
            grad_inputs = tf.gradients(loss, input1)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                data = np.random.rand(2, 2)
                output_value_ref = sess.run(output, feed_dict={input1: data})
                label_value = output_value_ref - 1.0
                grad_input_value_ref = sess.run(grad_inputs[0],
                                                feed_dict={input1: data,
                                                           label1: label_value})
                net = TFNet.from_session(sess, [input1], [output], generate_backward=True)

        output_value = net.forward(data)

        grad_input_value = net.backward(data, np.ones(shape=(2, 1)))

        self.assert_allclose(output_value, output_value_ref)
        self.assert_allclose(grad_input_value, grad_input_value_ref)

    def test_tf_optimizer_with_sparse_gradient(self):
        import tensorflow as tf

        ids = np.random.randint(0, 10, size=[40])
        labels = np.random.randint(0, 5, size=[40])
        id_rdd = self.sc.parallelize(ids)
        label_rdd = self.sc.parallelize(labels)
        training_rdd = id_rdd.zip(label_rdd).map(lambda x: [x[0], x[1]])
        with tf.Graph().as_default():
            dataset = TFDataset.from_rdd(training_rdd,
                                         names=["ids", "labels"],
                                         shapes=[[], []],
                                         types=[tf.int32, tf.int32],
                                         batch_size=8)
            id_tensor, label_tensor = dataset.tensors
            embedding_table = tf.get_variable(
                name="word_embedding",
                shape=[10, 5])

            embedding = tf.nn.embedding_lookup(embedding_table, id_tensor)
            loss = tf.reduce_mean(tf.losses.
                                  sparse_softmax_cross_entropy(logits=embedding,
                                                               labels=label_tensor))
            optimizer = TFOptimizer(loss, Adam(1e-3))
            optimizer.optimize(end_trigger=MaxEpoch(1))
            optimizer.sess.close()

    def test_tf_optimizer_with_sparse_gradient_using_keras(self):
        import tensorflow as tf

        ids = np.random.randint(0, 10, size=[40])
        labels = np.random.randint(0, 5, size=[40])
        id_rdd = self.sc.parallelize(ids)
        label_rdd = self.sc.parallelize(labels)
        training_rdd = id_rdd.zip(label_rdd).map(lambda x: [x[0], x[1]])
        with tf.Graph().as_default():
            dataset = TFDataset.from_rdd(training_rdd,
                                         names=["ids", "labels"],
                                         shapes=[[], []],
                                         types=[tf.int32, tf.int32],
                                         batch_size=8)
            from tensorflow.python.ops import variable_scope

            def variable_creator(**kwargs):
                kwargs["use_resource"] = False
                return variable_scope.default_variable_creator(None, **kwargs)

            getter = lambda next_creator, **kwargs: variable_creator(**kwargs)
            with variable_scope.variable_creator_scope(getter):
                words_input = tf.keras.layers.Input(shape=(), name='words_input')
                embedding_layer = tf.keras.layers.Embedding(input_dim=10,
                                                            output_dim=5, name='word_embedding')
                word_embeddings = embedding_layer(words_input)
                embedding = tf.keras.layers.Flatten()(word_embeddings)
                output = tf.keras.layers.Dense(5, activation="softmax")(embedding)
                model = tf.keras.models.Model(inputs=[words_input], outputs=[output])
                model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")

            optimizer = TFOptimizer.from_keras(model, dataset)
            optimizer.optimize(end_trigger=MaxEpoch(1))
            optimizer.sess.close()


if __name__ == "__main__":
    pytest.main([__file__])
