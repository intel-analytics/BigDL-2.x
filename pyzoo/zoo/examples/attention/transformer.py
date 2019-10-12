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


from bigdl.optim.optimizer import Adam
from keras.datasets import imdb
from keras.preprocessing import sequence
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras.layers import *
from zoo.common.nncontext import init_spark_conf
from zoo.common.nncontext import init_nncontext


conf = init_spark_conf()
conf.set("spark.executor.extraJavaOptions", "-Xss512m")
conf.set("spark.driver.extraJavaOptions", "-Xss512m")
sc = init_nncontext(conf)
max_features = 20000
max_len = 200

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


train_pos = np.zeros((len(x_train), max_len), dtype=np.int32)
val_pos = np.zeros((len(x_test), max_len), dtype=np.int32)
for i in range(0, len(x_train)):
    train_pos[i, :] = np.arange(max_len)
    val_pos[i, :] = np.arange(max_len)


def build_sample(token_id, position_id, label):
    samples = []
    for i in range(label.shape[0]):
        sample = Sample.from_ndarray([token_id[i], position_id[i]], np.array(label[i]))
        samples.append(sample)
    return samples


train_samples = build_sample(x_train, train_pos, y_train)
train_rdd = sc.parallelize(train_samples)
val_samples = build_sample(x_test, val_pos, y_test)
val_rdd = sc.parallelize(val_samples)

token_shape = (max_len,)
position_shape = (max_len,)
token_input = Input(shape=token_shape)
position_input = Input(shape=position_shape)
O_seq = TransformerLayer.init(
    vocab=max_features, hidden_size=128, n_head=8, seq_len=max_len)([token_input, position_input])
# Select the first output of the Transformer. The second is the pooled output.
O_seq = SelectTable(0)(O_seq)
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.2)(O_seq)
outputs = Dense(2, activation='softmax')(O_seq)

model = Model([token_input, position_input], outputs)
model.summary()

model.compile(optimizer=Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

batch_size = 160
print('Train...')
model.fit(train_rdd,
          batch_size=batch_size,
          nb_epoch=1)
print("Train finished.")

print('Evaluating...')
score = model.evaluate(val_rdd, batch_size=160)[0]
print(score)
