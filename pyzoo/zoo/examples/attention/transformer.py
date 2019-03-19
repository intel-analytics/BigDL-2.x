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
from zoo.pipeline.api.autograd import *
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

xmb = np.zeros((len(x_train), max_len, 2), dtype=np.int32)
# Position information that is added to the input embeddings in the TransformerModel
xmb[:, :, 1] = np.arange(max_len)
xmb[:, :, 0] = x_train

xmb_val = np.zeros((len(x_test), max_len, 2), dtype=np.int32)
# Position information that is added to the input embeddings in the TransformerModel
xmb_val[:, :, 1] = np.arange(max_len)
xmb_val[:, :, 0] = x_test
S_inputs = Input(shape=(max_len, 2))
O_seq = TransformerLayer.init_with_default_embedding(
    vocab=max_features, hidden_size=128, n_head=8, seq_len=max_len)(S_inputs)
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.2)(O_seq)
outputs = Dense(2, activation='softmax')(O_seq)


model = Model(S_inputs, outputs)
model.summary()

model.compile(optimizer=Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

batch_size = 160
print('Train...')
model.fit(xmb, y_train,
          batch_size=batch_size,
          nb_epoch=1)
print("Train finished.")

print('Evaluating...')
score = model.evaluate(xmb_val, y_test, batch_size=160)[0]
print(score)
