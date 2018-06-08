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

from zoo.pipeline.api.net import Net

import zoo.pipeline.api.autograd as A
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *

# export SPARK_DRIVER_MEMORY=20g


image_shape = [3, 224, 224]
input_shape = [2] + image_shape


def resnet50_features_extraction():
    resnet50 = Net.load_bigdl(
        "/home/lizhichao/bin/data/analytics-zoo_resnet-50_imagenet_0.1.0.model")
    features_extraction = Sequential()
    features_extraction.add(InputLayer(input_shape=image_shape))
    features_extraction.add(resnet50.new_graph(outputs=["pool5"]).to_keras())
    features_extraction.add(Flatten())  # shape is : 2048
    return features_extraction


input = Input(shape=input_shape)
features = TimeDistributed(layer=resnet50_features_extraction())(input)
f1 = features.index_select(1, 0)  # dim0 is batch, dim1 is time_steps
f2 = features.index_select(1, 1)
diff = A.abs(f1 - f2)
fc = Dense(1)(diff)
output = Activation("sigmoid")(fc)
model = Model(input, output)

model.compile(optimizer=SGD(learningrate=0.001),
              loss=BCECriterion())
sample_num = 24
mock_x = np.random.uniform(0, 1, [sample_num] + input_shape)
mock_y = np.random.randint(0, 1, sample_num)
model.fit(x=mock_x, y=mock_y, batch_size=12, nb_epoch=1)

out_data = model.forward(mock_x)
print(out_data.shape)
