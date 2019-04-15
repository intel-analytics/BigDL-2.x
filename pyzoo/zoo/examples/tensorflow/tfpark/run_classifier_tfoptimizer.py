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

from optparse import OptionParser

import tensorflow as tf
from bigdl.optim.optimizer import Adam, MaxEpoch
from zoo.common.nncontext import *
from zoo.pipeline.api.net import TFDataset, TFOptimizer, TFPredictor
from bert import tokenization
from bert.modeling import BertConfig, get_assignment_map_from_checkpoint, BertModel


# Copy code from BERT run_classifier.py since import run_classifier will have error in imports such as
# import modeling

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      import csv
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return loss, per_example_loss, logits, probabilities


def to_list_numpy(feature):
    return [np.array(feature.input_ids), np.array(feature.input_mask),
            np.array(feature.segment_ids), np.array(feature.label_id)]


def generate_tf_dataset(examples, seq_len, batch_size, mode="train"):
    features = convert_examples_to_features(examples, label_list, seq_len, tokenizer)
    features = [to_list_numpy(feature) for feature in features]
    rdd = sc.parallelize(features)
    if mode == "eval" or mode == "predict":
        return TFDataset.from_rdd(rdd,
                                  names=["input_ids", "input_mask", "segment_ids", "label_id"],
                                  shapes=[[seq_len], [seq_len], [seq_len], []],
                                  types=[tf.int32, tf.int32, tf.int32, tf.int32],
                                  batch_per_thread=4)
    else:
        return TFDataset.from_rdd(rdd,
                                  names=["input_ids", "input_mask", "segment_ids", "label_id"],
                                  shapes=[[seq_len], [seq_len], [seq_len], []],
                                  types=[tf.int32, tf.int32, tf.int32, tf.int32],
                                  batch_size=batch_size)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--bert_base_dir", dest="bert_base_dir")
    parser.add_option("--data_dir", dest="data_dir")
    parser.add_option("--output_dir", dest="output_dir")
    parser.add_option("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_option("--max_seq_length", dest="max_seq_length", type=int, default=128)
    parser.add_option("-e", "--nb_epoch", dest="nb_epoch", type=int, default=3)
    parser.add_option("-l", "--learning_rate", dest="learning_rate", type=float, default=2e-5)

    (options, args) = parser.parse_args(sys.argv)
    # Model and data files
    bert_config_file = options.bert_base_dir + "/bert_config.json"
    vocab_file = options.bert_base_dir + "/vocab.txt"
    init_checkpoint = options.bert_base_dir + "/bert_model.ckpt"
    bert_config = BertConfig.from_json_file(bert_config_file)

    sc = init_nncontext("BERT MRPC Classification Example")

    # Data preparation and preprocessing
    processor = MrpcProcessor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file)
    train_examples = processor.get_train_examples(options.data_dir)
    train_dataset = generate_tf_dataset(train_examples, options.max_seq_length, options.batch_size)
    eval_examples = processor.get_dev_examples(options.data_dir)
    eval_dataset = generate_tf_dataset(eval_examples, options.max_seq_length, options.batch_size, "eval")

    # Model loading and construction
    input_ids, input_mask, segment_ids, label_ids = train_dataset.tensors
    loss, per_example_loss, logits, probabilities = create_model(bert_config, True, input_ids, input_mask, segment_ids,
                                                                 label_ids, len(label_list), False)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint:
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # Training
    optimizer = TFOptimizer(loss, Adam(options.learning_rate))
    optimizer.optimize(end_trigger=MaxEpoch(options.nb_epoch))
    saver = tf.train.Saver()
    saver.save(optimizer.sess, options.output_dir + "/model")

    # Evaluation
    tf.reset_default_graph()
    sess = tf.Session()
    input_ids, input_mask, segment_ids, label_ids = eval_dataset.tensors
    loss, per_example_loss, logits, probabilities = create_model(bert_config, False, input_ids, input_mask, segment_ids,
                                                                 label_ids, len(label_list), False)
    saver = tf.train.Saver()
    saver.restore(sess, options.output_dir + "/model")
    predictions = tf.to_int32(tf.argmax(logits, axis=1))
    correct = tf.expand_dims(tf.to_int32(tf.equal(predictions, label_ids)), axis=1)
    predictor = TFPredictor(sess, [correct])
    accuracy = predictor.predict().mean()
    print("Evaluation accuracy is %s" % accuracy[0])
