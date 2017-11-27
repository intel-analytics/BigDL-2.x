#
# Copyright 2016 The BigDL Authors.
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
from __future__ import print_function

import pytest
from keras.layers import *

np.random.seed(1337)  # for reproducibility
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers import Dense, Input
from bigdl.keras.converter import *
from test.bigdl.test_utils import BigDLTestCase
from keras.regularizers import l1, l2, l1l2


class TestLayer(BigDLTestCase):

    def test_dense(self):
        input_data = np.random.random_sample([1, 10])
        layer = Dense(2, init='one', activation="relu",
                      input_shape=(10, ), W_regularizer=l1l2(l1=0.01, l2=0.02))
        self.modelTestSingleLayer(input_data, layer, dump_weights=True)
        layer2 = Dense(2, init='one', activation="softplus",
                       input_shape=(10, ), b_regularizer=l2(0.02))
        self.modelTestSingleLayer(input_data, layer2, dump_weights=True)
        layer3 = Dense(2, init='one', input_shape=(10, ),
                       W_regularizer=keras.regularizers.WeightRegularizer(l1=0.1))
        self.modelTestSingleLayer(input_data, layer3, dump_weights=True)
        layer4 = Dense(2, init='glorot_uniform', activation="hard_sigmoid", input_shape=(10, ))
        self.modelTestSingleLayer(input_data, layer4, dump_weights=True)

    def test_timedistributeddense(self):
        input_data = np.random.random_sample([2, 4, 5])
        layer = TimeDistributedDense(6, input_shape=(4, 5))
        self.modelTestSingleLayer(input_data, layer, dump_weights=True)

    def test_embedding(self):
        # Test index start from 0
        input_data = np.array([[0, 1, 2, 99], [0, 4, 5, 99]])
        layer = Embedding(input_dim=100,  # index [0,99]
                          output_dim=64,  # vector dim
                          init='uniform',
                          input_length=None,
                          W_regularizer=None, activity_regularizer=None,
                          W_constraint=None,
                          mask_zero=False,
                          weights=None, dropout=0.)
        self.modelTestSingleLayer(input_data, layer, dump_weights=True)
        # Random input
        input_data2 = np.random.randint(100, size=(10, 128))  # batch: 20, seqlen 128
        self.modelTestSingleLayer(input_data2, layer, dump_weights=True)

        # TODO: add test that exception would be raised if input_lenght == 6
        with pytest.raises(Exception) as excinfo:
            layer = Embedding(input_dim=100,  # index [0,99]
                              output_dim=64,  # vector dim
                              init='uniform',
                              input_length=111,
                              W_regularizer=None, activity_regularizer=None,
                              W_constraint=None,
                              mask_zero=False,
                              weights=None, dropout=0.)
            input_data = np.random.randint(100, size=(10, 128))  # batch: 20, seqlen 128
            self.modelTestSingleLayer(input_data, layer, dump_weights=True)
        assert str(excinfo.value) == """The input_length doesn't match: 128 vs 111"""

    # TODO: Add more test case? activation
    def test_conv1D(self):
        input_data = np.random.random_sample([1, 10, 32])
        layer = lambda: Convolution1D(64, 3, border_mode='valid', input_shape=(10, 32))
        self.modelTestSingleLayerWithOrdersModes(input_data, layer,
                                                 dump_weights=True, dim_orderings=["tf"])

        layer2 = lambda: Convolution1D(64, 3, border_mode='same',
                                       input_shape=(10, 32))
        self.modelTestSingleLayerWithOrdersModes(input_data, layer2,
                                                 dump_weights=True, dim_orderings=["tf"])

        layer3 = lambda: Convolution1D(64, 3, border_mode='same',
                                       activation="relu", input_shape=(10, 32))
        self.modelTestSingleLayerWithOrdersModes(input_data, layer3,
                                                 dump_weights=True, dim_orderings=["tf"])

    def _load_keras(self, json_path, hdf5_path):
        with open(json_path, "r") as jp:
            kmodel = model_from_json(jp.read())
        kmodel.load_weights_from_hdf5(hdf5_path)
        bmodel = DefinitionLoader.from_json_path(json_path)
        WeightLoader.load_weights_from_hdf5(bmodel, kmodel, hdf5_path)
        return kmodel, bmodel

    def test_conv2D(self):
        input_data = np.random.random_sample([1, 3, 128, 128])
        layer = lambda: Convolution2D(64, 1, 20, input_shape=(3, 128, 128))
        self.modelTestSingleLayerWithOrdersModes(input_data,
                                                 layer,
                                                 dump_weights=True, rtol=1e-5, atol=1e-5)
        # Test if alias works or not
        layer = lambda: Conv2D(64, 3, 1, input_shape=(3, 128, 128))
        self.modelTestSingleLayerWithOrdersModes(input_data,
                                                 layer,
                                                 dump_weights=True, rtol=1e-5, atol=1e-5)

    def test_conv3D(self):
        input_data = np.random.random_sample([1, 3, 32, 32, 32])
        layer = Convolution3D(12, 5, 3, 4, dim_ordering="th",
                              border_mode="valid", subsample=(1, 1, 2),
                              input_shape=(3, 32, 32, 32))
        self.modelTestSingleLayer(input_data, layer,
                                  dump_weights=True, rtol=1e-5, atol=1e-5)

    def test_atrousconvolution1d(self):
        input_data = np.random.random_sample([2, 10, 32])
        layer = AtrousConvolution1D(64, 3, atrous_rate=2,
                                    border_mode="valid", activation='relu',
                                    input_shape=(10, 32))
        self.modelTestSingleLayer(input_data, layer, dump_weights=True)

    def test_atrousconvolution2d(self):
        input_data = np.random.random_sample([1, 3, 128, 128])
        layer = AtrousConvolution2D(64, 3, 4, atrous_rate=(2, 2), dim_ordering="th",
                                    border_mode="valid", activation='tanh',
                                    input_shape=(3, 128, 128))
        self.modelTestSingleLayer(input_data, layer, dump_weights=True)

    def test_deconvolution2d(self):
        input_data = np.random.random_sample([32, 3, 12, 12])
        layer = Deconvolution2D(3, 3, 3, output_shape=(None, 3, 14, 14),
                                border_mode="valid", dim_ordering="th",
                                input_shape=(3, 12, 12))
        self.modelTestSingleLayer(input_data, layer, dump_weights=True)
        layer2 = Deconvolution2D(3, 3, 3, output_shape=(None, 3, 25, 25),
                                 border_mode="valid", subsample=(2, 2),
                                 dim_ordering="th", input_shape=(3, 12, 12))
        self.modelTestSingleLayer(input_data, layer2, dump_weights=True)

    def test_maxpooling3d(self):
        input_data = np.random.random_sample([1, 3, 20, 15, 35])
        layer = MaxPooling3D(pool_size=(2, 2, 4), strides=(3, 1, 5), dim_ordering="th",
                             border_mode="valid", input_shape=(3, 20, 15, 35))
        self.modelTestSingleLayer(input_data, layer)

    def test_maxpooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = lambda: MaxPooling2D(pool_size=[2, 3], strides=[4, 2],
                                     border_mode="valid", input_shape=(3, 20, 20))
        self.modelTestSingleLayerWithOrdersModes(input_data, layer)
        layer2 = lambda: MaxPooling2D(pool_size=[1, 1], strides=[2, 2],
                                      border_mode="valid", input_shape=(3, 20, 20))
        self.modelTestSingleLayerWithOrdersModes(input_data, layer2)

    def test_maxpooling1d(self):
        input_data = np.random.random_sample([5, 96, 64])
        layer = MaxPooling1D(pool_length=4, stride=None,
                             border_mode='valid', input_shape=(96, 64))
        self.modelTestSingleLayer(input_data, layer)
        input_data2 = np.random.random_sample([1, 3, 20])
        layer2 = MaxPooling1D(pool_length=2, stride=None,
                              border_mode='valid', input_shape=(3, 20))
        self.modelTestSingleLayer(input_data2, layer2)

    def test_globalmaxpooling3d(self):
        input_data = np.random.random_sample([1, 5, 20, 25, 35])
        layer = GlobalMaxPooling3D(dim_ordering="th", input_shape=(5, 20, 25, 35))
        self.modelTestSingleLayer(input_data, layer)

    def test_globalmaxpooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = lambda: GlobalMaxPooling2D(input_shape=(3, 20, 20))
        self.modelTestSingleLayerWithOrdersModes(input_data, layer,
                                                 border_modes=[None])

    def test_globalmaxpooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = GlobalMaxPooling1D(input_shape=(3, 20))
        self.modelTestSingleLayer(input_data, layer)

    def test_averagepooling3d(self):
        input_data = np.random.random_sample([2, 6, 20, 15, 35])
        layer = AveragePooling3D(pool_size=(2, 3, 4), strides=(3, 1, 5), dim_ordering="th",
                                 border_mode="valid", input_shape=(3, 20, 15, 35))
        self.modelTestSingleLayer(input_data, layer)

    def test_averagepooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = lambda: AveragePooling2D(pool_size=[2, 3], strides=[4, 2],
                                         border_mode="valid", input_shape=(3, 20, 20))
        self.modelTestSingleLayerWithOrdersModes(input_data, layer)
        layer2 = lambda: AveragePooling2D(pool_size=[1, 1], strides=[2, 2],
                                          border_mode="valid", input_shape=(3, 20, 20))
        self.modelTestSingleLayerWithOrdersModes(input_data, layer2)

    def test_averagepooling1d(self):
        input_data = np.random.random_sample([5, 96, 64])
        layer = lambda: AveragePooling1D(pool_length=4, stride=None,
                                         border_mode='valid', input_shape=(96, 64))
        self.modelTestSingleLayerWithOrdersModes(input_data, layer, dim_orderings=["tf"])

    def test_globalaveragepooling3d(self):
        input_data = np.random.random_sample([1, 5, 20, 25, 35])
        layer = GlobalAveragePooling3D(dim_ordering="th", input_shape=(5, 20, 25, 35))
        self.modelTestSingleLayer(input_data, layer, rtol=1e-5, atol=1e-5)

    def test_globalaveragepooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = lambda: GlobalAveragePooling2D(input_shape=(3, 20, 20))
        self.modelTestSingleLayerWithOrdersModes(input_data, layer,
                                                 border_modes=[None])

    def test_globalaveragepooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = GlobalAveragePooling1D(input_shape=(3, 20))
        self.modelTestSingleLayer(input_data, layer)

    def test_batchnormalization(self):
        input_data = np.random.random_sample([2, 6, 128, 128])
        layer = BatchNormalization(input_shape=(6, 128, 128), axis=1)
        self.modelTestSingleLayer(input_data, layer, dump_weights=True, random_weights=False)

    def test_flatten(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = Flatten(input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer)

    def test_permute(self):
        input_data = np.random.random_sample([5, 4, 3, 2, 6])
        layer1 = Permute((4, 1, 2, 3), input_shape=(4, 3, 2, 6))
        self.modelTestSingleLayer(input_data, layer1)
        layer2 = Permute((4, 3, 2, 1), input_shape=(4, 3, 2, 6))
        self.modelTestSingleLayer(input_data, layer2)
        layer3 = Permute((1, 2, 3, 4), input_shape=(4, 3, 2, 6))
        self.modelTestSingleLayer(input_data, layer3)
        layer4 = Permute((1, 4, 2, 3), input_shape=(4, 3, 2, 6))
        self.modelTestSingleLayer(input_data, layer4)

    def test_reshape(self):
        input_data = np.random.random_sample([1, 3, 5, 4])
        layer = Reshape(target_shape=(3, 20), input_shape=(3, 5, 4))
        self.modelTestSingleLayer(input_data, layer)

    def test_repeatvector(self):
        input_data = np.random.random_sample([2, 3])
        layer = RepeatVector(4, input_shape=(3, ))
        self.modelTestSingleLayer(input_data, layer)

    def test_nested_model_seq_concat(self):
        input_data1 = np.random.random_sample([2, 3])
        input1 = Input((3,))
        out1 = Dense(3)(input1)
        out1_1 = Dense(3)(out1)

        branch1 = Model(input=[input1], output=out1_1)

        branch2 = Sequential()
        branch2.add(Dense(3, input_shape=[3]))
        branch2.add(Dense(3))
        branch2.add(branch1)
        branch2_tensor = branch2(input1)

        kmodel = Model(input=[input1], output=branch2_tensor)
        kmodel.predict([input_data1])
        self.modelTest(input_data1,
                       kmodel,
                       random_weights=False,
                       dump_weights=True,
                       is_training=False)

    def test_merge_method_concat(self):
        input_data1 = np.random.random_sample([2, 4])
        input_data2 = np.random.random_sample([2, 3])
        input1 = Input((4,))
        input2 = Input((3,))
        out1 = Dense(4)(input1)
        out2 = Dense(3)(input2)
        from keras.engine import merge
        m = merge([out1, out2], mode="concat", concat_axis=1)
        kmodel = Model(input=[input1, input2], output=m)

        self.modelTest([input_data1, input_data2],
                       kmodel,
                       random_weights=False,
                       dump_weights=True,
                       is_training=False)

    def test_merge_method_mix_concat(self):
        input_data1 = np.random.random_sample([2, 4])
        input_data2 = np.random.random_sample([2, 3])
        input1 = Input((4,))
        input2 = Input((3,))
        out1 = Dense(4)(input1)
        branch1 = Model(input1, out1)(input1)
        branch2 = Dense(3)(input2)
        from keras.engine import merge
        m = merge([branch1, branch2], mode="concat", concat_axis=1)
        kmodel = Model(input=[input1, input2], output=m)

        self.modelTest([input_data1, input_data2],
                       kmodel,
                       random_weights=False,
                       dump_weights=True,
                       is_training=False)

    def test_merge_model_seq_concat(self):
        input_data1 = np.random.random_sample([2, 4])
        input_data2 = np.random.random_sample([2, 3])
        input1 = Input((4,))
        input2 = Input((3,))
        out1 = Dense(4)(input1)
        out1_1 = Dense(4)(out1)

        branch1 = Model(input=[input1], output=out1_1)
        branch2 = Sequential()
        branch2.add(Dense(3, input_shape=[3]))
        branch2.add(Dense(3))
        branch1_tensor = branch1(input1)
        branch2_tensor = branch2(input2)

        from keras.engine import merge
        m = merge([branch1_tensor, branch2_tensor], mode="concat", concat_axis=1)
        kmodel = Model(input=[input1, input2], output=m)
        kmodel.predict([input_data1, input_data2])
        self.modelTest([input_data1, input_data2],
                       kmodel,
                       random_weights=False,
                       dump_weights=True,
                       is_training=False)

    def test_merge_model_model_concat(self):
        input_data1 = np.random.random_sample([2, 4])
        input_data2 = np.random.random_sample([2, 3])
        input1 = Input((4,))
        input2 = Input((3,))
        out1 = Dense(4)(input1)
        out1_1 = Dense(4)(out1)

        out2 = Dense(3)(input2)
        out2_1 = Dense(3)(out2)

        branch1 = Model(input=[input1], output=out1_1)
        branch2 = Model(input=[input2], output=out2_1)
        branch1_tensor = branch1(input1)
        branch2_tensor = branch2(input2)

        from keras.engine import merge
        m = merge([branch1_tensor, branch2_tensor], mode="concat", concat_axis=1)
        kmodel = Model(input=[input1, input2], output=m)

        self.modelTest([input_data1, input_data2],
                       kmodel,
                       random_weights=False,
                       dump_weights=True,
                       is_training=False)

    def test_merge_seq_seq_concat(self):
        input_data1 = np.random.random_sample([2, 4])
        input_data2 = np.random.random_sample([2, 3])
        branch1 = Sequential()
        branch1.add(Dense(20, input_shape=[4]))

        branch2 = Sequential()
        branch2.add(Dense(10, input_shape=[3]))

        merged_model = Sequential()
        merged_model.add(Merge([branch1, branch2], mode='concat', concat_axis=1))

        self.modelTestSingleLayer([input_data1, input_data2],
                                  Merge([branch1, branch2], mode='concat', concat_axis=1),
                                  dump_weights=True,
                                  functional_apis=[False])

    def test_merge_concat(self):
        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 8))
        inputLayer3 = InputLayer(input_shape=(3, 6, 9))

        layer = Merge([inputLayer1, inputLayer2, inputLayer3], mode='concat', concat_axis=3)
        # the index including batch and start from zero which is the index to be merge
        input_data = [np.random.random_sample([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 8]),
                      np.random.random([2, 3, 6, 9])]
        self.modelTestSingleLayer(input_data, layer, functional_apis=[False])

    def test_merge_sum(self):
        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 7))
        inputLayer3 = InputLayer(input_shape=(3, 6, 7))

        layer = Merge([inputLayer1, inputLayer2, inputLayer3], mode='sum')
        # the index including batch and start from zero, and it's the index to be merge
        input_data = [np.random.random_sample([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7])]
        self.modelTestSingleLayer(input_data, layer, functional_apis=[False])

    def test_merge_mul(self):
        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 7))
        inputLayer3 = InputLayer(input_shape=(3, 6, 7))

        layer = Merge([inputLayer1, inputLayer2, inputLayer3], mode='mul')
        input_data = [np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7])]
        self.modelTestSingleLayer(input_data, layer, functional_apis=[False])

    def test_merge_max(self):
        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 7))
        inputLayer3 = InputLayer(input_shape=(3, 6, 7))

        layer = Merge([inputLayer1, inputLayer2, inputLayer3], mode='max')
        input_data = [np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7])]
        self.modelTestSingleLayer(input_data, layer, functional_apis=[False])

    def test_merge_ave(self):
        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 7))
        inputLayer3 = InputLayer(input_shape=(3, 6, 7))

        layer = Merge([inputLayer1, inputLayer2, inputLayer3], mode='ave')
        input_data = [np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7])]
        self.modelTestSingleLayer(input_data, layer, functional_apis=[False])

    def test_merge_dot(self):
        inputLayer1 = InputLayer(input_shape=(3, ))
        inputLayer2 = InputLayer(input_shape=(3, ))

        layer = Merge([inputLayer1, inputLayer2], mode='dot')
        input_data = [np.random.random([2, 3]),
                      np.random.random([2, 3])]
        self.modelTestSingleLayer(input_data, layer, functional_apis=[False])

    def test_elu(self):
        input_data = np.random.random_sample([10, 2, 3, 4])
        layer = ELU(alpha=1.0, input_shape=(2, 3, 4))
        self.modelTestSingleLayer(input_data, layer)

    def test_prelu(self):
        input_data = np.random.random_sample([1, 2, 3, 4])
        layer = PReLU(input_shape=(2, 3, 4))
        self.modelTestSingleLayer(input_data, layer)

    def test_leakyrelu(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = LeakyReLU(alpha=0.5, input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer)

    def test_thresholdedrelu(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = ThresholdedReLU(theta=0.2, input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer)

    def test_parametricsoftplus(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = ParametricSoftplus(alpha_init=0.4, beta_init=2.5, input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer, random_weights=False)

    def test_zeropadding1d(self):
        input_data = np.random.uniform(0, 1, [3, 2, 3])
        layer1 = ZeroPadding1D(padding=3, input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer1)
        layer2 = ZeroPadding1D(padding=(2, 3), input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer2)
        layer3 = ZeroPadding1D(padding={'left_pad': 1, 'right_pad': 2}, input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer3)

    def test_zeropadding2d(self):
        input_data = np.random.uniform(0, 1, [1, 2, 3, 4])
        layer1 = ZeroPadding2D(padding=(2, 3), input_shape=(2, 3, 4))
        self.modelTestSingleLayer(input_data, layer1)
        layer2 = ZeroPadding2D(padding=(2, 3, 4, 1), input_shape=(2, 3, 4))
        self.modelTestSingleLayer(input_data, layer2)
        layer3 = ZeroPadding2D(
            padding={'top_pad': 1, 'bottom_pad': 2, 'left_pad': 3, 'right_pad': 4},
            input_shape=(2, 3, 4))
        self.modelTestSingleLayer(input_data, layer3)

    def test_zeropadding3d(self):
        input_data = np.random.uniform(0, 1, [3, 2, 4, 1, 5])
        layer = ZeroPadding3D(padding=(1, 2, 3), input_shape=(2, 4, 1, 5))
        self.modelTestSingleLayer(input_data, layer)

    def test_cropping1d(self):
        input_data = np.random.uniform(0, 1, [3, 10, 10])
        layer = Cropping1D(cropping=(1, 2))
        self.modelTestSingleLayer(input_data, layer)

    def test_simplernn(self):
        input_data = np.random.random([3, 4, 5])
        layer = SimpleRNN(5, input_shape=(4, 5), return_sequences=True)
        self.modelTestSingleLayer(input_data, layer, dump_weights=True)
        layer2 = SimpleRNN(3, input_shape=(4, 5), return_sequences=False)
        self.modelTestSingleLayer(input_data, layer2, dump_weights=True)
        layer3 = SimpleRNN(3, input_shape=(4, 5), activation='relu')
        self.modelTestSingleLayer(input_data, layer3, dump_weights=True)

    def test_lstm(self):
        input_data = np.random.random([3, 4, 5])
        layer = LSTM(5, input_shape=(4, 5), return_sequences=True)
        self.modelTestSingleLayer(input_data, layer, dump_weights=True)
        layer2 = LSTM(3, input_shape=(4, 5), return_sequences=False,
                      activation='relu', inner_activation='sigmoid')
        self.modelTestSingleLayer(input_data, layer2, dump_weights=True)

    def test_convlstm2d(self):
        input_data = np.random.random_sample([4, 8, 40, 40, 32])
        layer = ConvLSTM2D(32, 4, 4, input_shape=(8, 40, 40, 32),
                           return_sequences=True, border_mode='same')
        self.modelTestSingleLayer(input_data, layer, dump_weights=True, random_weights=True)
        layer2 = ConvLSTM2D(32, 4, 4, input_shape=(8, 40, 40, 32), return_sequences=True,
                            activation='relu', inner_activation='sigmoid', border_mode='same')
        self.modelTestSingleLayer(input_data, layer2, dump_weights=True,
                                  random_weights=True, rtol=1e-5, atol=1e-5)

    def test_gru(self):
        input_data = np.random.random([3, 4, 5])
        layer = GRU(4, input_shape=(4, 5), return_sequences=True)
        self.modelTestSingleLayer(input_data, layer, dump_weights=True)
        layer2 = GRU(8, input_shape=(4, 5), return_sequences=False,
                     activation='relu', inner_activation='sigmoid')
        self.modelTestSingleLayer(input_data, layer2, dump_weights=True)

    # TODO: Support share weights training.
    def test_multiple_inputs_share_weights(self):
        with pytest.raises(Exception) as excinfo:
            input_node1 = Input(shape=[3, 16, 16])
            input_node2 = Input(shape=[3, 32, 32])
            conv2d = Convolution2D(5, 3, 3,
                                   border_mode='same')
            conv1 = conv2d(input_node1)
            conv2 = conv2d(input_node2)
            out1 = Flatten()(conv1)
            out2 = Flatten()(conv2)
            model1 = Model(input=[input_node1, input_node2], output=[out1, out2])
            tensor1, tensor2 = model1([input_node1, input_node2])
            out3 = Dense(7)(tensor1)
            out4 = Dense(8)(tensor2)
            model2 = Model(input=[input_node1, input_node2], output=[out3, out4])
            def_path, w_path = self._dump_keras(model2)
            bigdl_model = DefinitionLoader.from_json_path(def_path)
        assert str(excinfo.value) == """Convolution2D doesn't support multiple inputs with shared weights"""  # noqa


if __name__ == "__main__":
    pytest.main([__file__])
