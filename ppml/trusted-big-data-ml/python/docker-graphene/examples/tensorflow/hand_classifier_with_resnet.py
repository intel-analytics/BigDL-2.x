"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

import tempfile

import sys
sys.path.append(".")

from work.tensorflow.resnets_utils import *
# from resnets_utils import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class hand_classifier(object):

    def __init__(self, model_save_path='./model_saving/hand_classifier'):
        self.model_save_path = model_save_path


    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        """
        Implementation of the identity block as defined in Figure 3

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            #first
            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #second
            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #third

            W_conv3 = self.weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            #final step
            add = tf.add(X, X_shortcut)
            add_result = tf.nn.relu(add)

        return add_result


    def convolutional_block(self, X_input, kernel_size, in_filter,
                            out_filters, stage, block, training, stride=2):
        """
        Implementation of the convolutional block as defined in Figure 4

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            #first
            W_conv1 = self.weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #second
            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #third
            W_conv3 = self.weight_variable([1,1, f2,f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            #shortcut path
            W_shortcut = self.weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            #final
            add = tf.add(x_shortcut, X)
            add_result = tf.nn.relu(add)

        return add_result


    def deepnn(self, x_input, classes=6):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:

        Returns:
        """
        x = tf.pad(x_input, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
        with tf.variable_scope('reference') :
            training = tf.placeholder(tf.bool, name='training')

            #stage 1
            w_conv1 = self.weight_variable([7, 7, 3, 64])
            x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
            x = tf.layers.batch_normalization(x, axis=3, training=training)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='VALID')
            print(x.shape)
            # assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))

            #stage 2
            x = self.convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', training, stride=1)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b', training=training)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)

            #stage 3
            x = self.convolutional_block(x, 3, 256, [128,128,512], 3, 'a', training)
            x = self.identity_block(x, 3, 512, [128,128,512], 3, 'b', training=training)
            x = self.identity_block(x, 3, 512, [128,128,512], 3, 'c', training=training)
            x = self.identity_block(x, 3, 512, [128,128,512], 3, 'd', training=training)

            #stage 4
            x = self.convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
            x = self.identity_block (x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)

            #stage 5
            x = self.convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training=training)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training=training)

            x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1,1,1,1], padding='VALID')

            flatten = tf.layers.flatten(x)
            x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
            # Dropout - controls the complexity of the model, prevents co-adaptation of
            # features.
            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                x = tf.nn.dropout(x, keep_prob)

            logits = tf.layers.dense(x, units=6, activation=tf.nn.softmax)

        return logits, keep_prob, training



    def deepnn0(self, x):
        """deepnn builds the graph for a deep net for classifying digits.
        Args:
          x: an input tensor with the dimensions (N_examples, 784), where 784 is the
          number of pixels in a standard MNIST image.
        Returns:
          A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
          equal to the logits of classifying the digit into one of 10 classes (the
          digits 0-9). keep_prob is a scalar placeholder for the probability of
          dropout.
          :param x:
          :param training:
          :return:
        """
        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
        with tf.name_scope('reshape'):
            x_image = tf.reshape(x, [-1, 64, 64, 3])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            W_conv1 = self.weight_variable([5, 5, 3, 32])
            # b_conv1 = bias_variable([32])

            training = tf.placeholder(tf.bool, name='training')
            conv1 = self.conv2d(x_image, W_conv1)
            batch_norm1 = tf.layers.batch_normalization(conv1, axis=3, training=training)

            # h_conv1 = tf.nn.relu(batch_norm1 + b_conv1)
            h_conv1 = tf.nn.relu(batch_norm1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = self.max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            W_conv2 = self.weight_variable([5, 5, 32, 64])
            # b_conv2 = bias_variable([64])
            conv2 = self.conv2d(h_pool1, W_conv2)
            batch_norm2 = tf.layers.batch_normalization(conv2, axis=3, training=training)
            h_conv2 = tf.nn.relu(batch_norm2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = self.max_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            W_fc1 = self.weight_variable([16 * 16 * 64, 1024])
            b_fc1 = self.bias_variable([1, 1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            W_fc2 = self.weight_variable([1024, 6])
            b_fc2 = self.bias_variable([1, 6])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return y_conv, keep_prob, training


    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost


    def accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy_op = tf.reduce_mean(correct_prediction)
        return accuracy_op

    def train(self, X_train, Y_train):
        features = tf.placeholder(tf.float32, [None, 64, 64, 3])
        labels = tf.placeholder(tf.int64, [None, 6])

        logits, keep_prob, train_mode = self.deepnn(features)

        cross_entropy = self.cost(logits, labels)

        with tf.name_scope('adam_optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        graph_location = tempfile.mkdtemp()
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())

        mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=32, seed=None)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                X_mini_batch, Y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]
                train_step.run(feed_dict={features: X_mini_batch, labels: Y_mini_batch, keep_prob: 0.5, train_mode: True})

                if i % 20 == 0:
                    train_cost = sess.run(cross_entropy, feed_dict={features: X_mini_batch,
                                          labels: Y_mini_batch, keep_prob: 1.0, train_mode: False})
                    print('step %d, training cost %g' % (i, train_cost))

            saver.save(sess, self.model_save_path)


    def evaluate(self, test_features, test_labels, name='test '):
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        y_ = tf.placeholder(tf.int64, [None, 6])

        logits, keep_prob, train_mode = self.deepnn(x)
        accuracy = self.accuracy(logits, y_)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_save_path)
            accu = sess.run(accuracy, feed_dict={x: test_features, y_: test_labels,keep_prob: 1.0, train_mode: False})
            print('%s accuracy %g' % (name, accu))


def main(_):
    data_dir = '/ppml/trusted-big-data-ml/work/tensorflow/Training-Data'
    # data_dir = 'Training-Data'
    print("## Beginning of Opening the File ##")
    orig_data = load_dataset(data_dir)
    print("## End of Opening the File ##")
    X_train, Y_train, X_test, Y_test = process_orig_datasets(orig_data)


    model = hand_classifier()
    model.train(X_train, Y_train)
    model.evaluate(X_test, Y_test)
    model.evaluate(X_train, Y_train, 'training data')




if __name__ == '__main__':
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    tf.app.run(main=main)
