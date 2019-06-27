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

# TODO: haven't finished yet
class ArrayLikeDataset(RayDataSet):
    def __init__(self, ins, batch_size, repeat=False, shuffle=False):
        # check ins is a list of Array-like with (shape)
        self.sample_num = ins[0].shape[0]
        self.repeat = repeat
        self.batch_size = batch_size
        self.ins = ins
        indexes = np.arange(self.sample_num)
        # batches = self.make_batch_indexes(self.sample_num, batch_size)
        self.next_start = 0
        if shuffle:
            np.random.shuffle(indexes)

    def next_batch(self):
        start, end = self.get_next_batch_index()
        return slice(self.ins, start, end)

    def slice(X, start=None, stop=None):
        if isinstance(X, list):
            return [x[start:stop] for x in X]
        else:
            return X[start:stop]

    def has_next(self):
        if self.next_start == self.sample_num and not self.repeat:
            return False
        if self.next_start == self.sample_num:
            self.next_start = 0
        return True

    def get_next_batch_index(self):
        if not self.has_next():
            raise Exception("End of data sequence")
        end = min(self.sample_num, self.next_start + self.batch_size)
        self.next_start = end
        return self.next_start, self.next_start + self.batch_size

    # def make_batch_indexes(self, size, batch_size):
    #     """Returns a list of batch indices (tuples of indices).
    #     """
    #     nb_batch = int(np.ceil(size / float(batch_size)))
    #     return [(i * batch_size, min(size, (i + 1) * batch_size))
    #             for i in range(0, nb_batch)]

class TFDataSetWrapper(RayDataSet):
    def __init__(self, input_fn, batch_size, repeat=True, shuffle=True):
        self.input_fn = input_fn
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle=shuffle
        self.init = False

    def action(self):
        if self.init:
            return
        self.init = True
        self.tf_dataset = self.input_fn()
        self.tf_dataset = self.tf_dataset.batch(self.batch_size)
        if self.shuffle:
            self.tf_dataset.shuffle(buffer_size=4 * self.batch_size)
        if self.repeat:
            self.tf_dataset = self.tf_dataset.repeat()
        self.session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=22, inter_op_parallelism_threads=22))
        self.x, self.y = self.tf_dataset.make_one_shot_iterator().get_next()

    def next_batch(self):
        if not self.init:
            raise Exception("Please invoke init() first")
        return [i for i in self.session.run([self.x, self.y])]

    def get_batchsize(self):
        return self.batch_size

