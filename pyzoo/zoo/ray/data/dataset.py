import numpy as np
import tensorflow as tf

class RayDataSet(object):
    def next_batch(self):
        raise Exception("not implemented")

    def get_batchsize(self):
        raise Exception("not implemented")

    @staticmethod
    def from_input_fn(input_fn, batch_size, repeat=True, shuffle=True):
        return TFDataSetWrapper(input_fn=input_fn, batch_size=batch_size, repeat=repeat, shuffle=shuffle)

class TFDataSetWrapper(RayDataSet):
    def __init__(self, input_fn, batch_size, repeat=True, shuffle=True):
        self.input_fn = input_fn
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle=shuffle
        self.tf_dataset = None
        self.session = None
        self.data_tmp = None

    def has_next(self):
        try:
            if not self.tf_dataset:
                print("Preparing for the dataset")
                self.tf_dataset = self.input_fn()
                self.tf_dataset = self.tf_dataset.batch(self.batch_size)
                if self.shuffle:
                    self.tf_dataset.shuffle(buffer_size=16 * self.batch_size)
                if self.repeat:
                    self.tf_dataset = self.tf_dataset.repeat()
                print("create session")
                self.session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                    intra_op_parallelism_threads=22, inter_op_parallelism_threads=22))
                print("getnext")
                self.x, self.y = self.tf_dataset.make_one_shot_iterator().get_next()
                print("End of prepare")

            self.data_tmp = [i for i in self.session.run([self.x, self.y])]
            return True
        except:
            self.data_tmp = None
            return False

    def next_batch(self):
        if self.data_tmp:
            return self.data_tmp
        else:
            raise Exception("End of data sequence")

    def get_batchsize(self):
        return self.batch_size


class DummyRayDataSet(RayDataSet):
    def __init__(self, feature_shape, label_shape):
        self.feature_shape=feature_shape
        self.label_shape=label_shape

    # it should return list of inputs and list of labels
    def next_batch(self):
        return [np.random.uniform(0, 1, size=self.feature_shape)], [np.random.uniform(0, 1, size=self.label_shape)]