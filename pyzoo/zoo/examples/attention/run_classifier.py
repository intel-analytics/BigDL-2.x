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
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.autograd import *
from zoo.common.nncontext import init_spark_conf
from zoo.common.nncontext import init_nncontext
from zoo.examples.attention.input import *

conf = init_spark_conf()
conf.set("spark.executor.extraJavaOptions", "-Xss512m")
conf.set("spark.driver.extraJavaOptions", "-Xss512m")
sc = init_nncontext(conf)

# bert_base_dir = "/home/ding/pretrain_model/bert/google/uncased_L-12_H-768_A-12"
# data_dir = "/home/ding/data/mrpc/glue_data/MRPC"
bert_base_dir = "/root/dingding/bert/uncased_L-12_H-768_A-12"
data_dir = "/root/dingding/bert/MRPC"
max_seq_length = 64
print('Loading data...')
processor = MrpcProcessor()
label_list = processor.get_labels()
tokenizer = tokenization.FullTokenizer(bert_base_dir + "/vocab.txt")

def build_sample(feature):
    sample = Sample.from_ndarray([feature.input_ids, feature.segment_ids, feature.pos_ids, feature.input_mask],
                                     feature.label_id)
    return sample


train_examples = processor.get_train_examples(data_dir)
features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
train_samples = [build_sample(feature) for feature in features]

# eval_examples = processor.get_dev_examples(options.data_dir)
# eval_rdd = generate_input_rdd(eval_examples, label_list, max_seq_length, tokenizer, "eval")
#
# test_examples = processor.get_test_examples(options.data_dir)
# test_rdd = generate_input_rdd(test_examples, label_list, max_seq_length, tokenizer, "test")

num_labels = 2
max_features = 50000
hidden_size = 128

train_data = sc.parallelize(train_samples)

token_shape = (max_seq_length,)
position_shape = (max_seq_length,)
segment_shape = (max_seq_length,)
mask_shape = (1, 1, max_seq_length)

token_input = Input(shape=token_shape)
position_input = Input(shape=position_shape)
segment_input = Input(shape=segment_shape)
mask_input = Input(shape=mask_shape)
bert = BERT.init(
    vocab=max_features, hidden_size=12, n_head=2, seq_len=max_seq_length)
O_seq = bert([token_input, segment_input, position_input, mask_input])
O_seq = SelectTable(bert.n_block)(O_seq)
O_seq = Dropout(0.2)(O_seq)
outputs = Dense(num_labels)(O_seq)

model = Model([token_input, segment_input, position_input, mask_input], outputs)
model.summary()

# t1=np.random.randint(4, size=(1, max_seq_length))
# t2=np.random.randint(1, size=(1, max_seq_length))
# t3=np.random.randint(max_seq_length-1, size=(1, 1, 1, max_seq_length))
# t4=np.random.randint(1, size=(1, max_seq_length))
# output = model.forward([t1, t2, t4, t3])

model.compile(optimizer=Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

batch_size = 160
# batch_size = 4
print('Train...')
model.fit(train_data,
          batch_size=batch_size,
          nb_epoch=1)
print("Train finished.")

# print('Evaluating...')
# score = model.evaluate(xmb_val, batch_size=160)[0]
# print(score)
