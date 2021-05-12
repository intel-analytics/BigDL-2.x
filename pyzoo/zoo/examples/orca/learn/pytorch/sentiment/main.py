
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
# ==============================================================================
# Most of the pytorch code is adapted from
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.learn.trigger import EveryEpoch


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The cluster mode, such as local, yarn or k8s.')
parser.add_argument('--backend', type=str, default="bigdl",
                    help='The backend of PyTorch Estimator; '
                         'bigdl and torch_distributed are supported')
args = parser.parse_args()

if args.cluster_mode == "local":
    init_orca_context(memory="4g")
elif args.cluster_mode == "yarn":
    init_orca_context(
        cluster_mode="yarn-client", num_nodes=2, driver_memory="4g",
        conf={"spark.rpc.message.maxSize": "1024",
              "spark.task.maxFailures": "1",
              "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})


def load_dataset(iter="None"):
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.

    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.

    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.

    """

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True,
                      fix_length=200)
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    # print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    # print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    # print("Label Length: " + str(len(LABEL.vocab)))

    if iter == "None":
        vocab_size = len(TEXT.vocab)
        return TEXT, vocab_size, word_embeddings
    else:
        train_data, valid_data = train_data.split()  # Further splitting of training_data to create new training_data & validation_data
        train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32,
                                                        sort_key=lambda x: len(x.text), repeat=False,
                                                        shuffle=True)
        if iter == "Train":
            return train_iter
        elif iter == "Test":
            return test_iter


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(LSTMClassifier, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights,
                                                   requires_grad=False)  # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        input = self.word_embeddings(
            input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))  # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))  # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[
                                      -1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)

        return final_output




TEXT, vocab_size, word_embeddings = load_dataset()

def model_creator(config):
    batch_size = 32
    output_size = 2
    hidden_size = 256
    embedding_length = 300
    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    return model

def optim_creator(model, config):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    return optim


def train_loader_creator(config, batch_size):
    train_iter = load_dataset(iter="Train")
    train_iter = ([batch.text[0], torch.autograd.Variable(batch.label).long()]
                 for batch in train_iter if batch.text[0].size()[0]==batch_size)
    return train_iter


def test_loader_creator(config, batch_size):
    test_iter = load_dataset(iter="Test")
    test_iter = ([batch.text[0], torch.autograd.Variable(batch.label).long()]
                for batch in test_iter if batch.text[0].size()[0]==batch_size)
    return test_iter

criterion = F.cross_entropy
batch_size = 32

train_iter = train_loader_creator({}, batch_size)
test_iter = test_loader_creator({}, batch_size)


if args.backend == "bigdl":
    net = model_creator({})
    optimizer = optim_creator(model=net,config={})
    orca_estimator = Estimator.from_torch(model=net,
                                          optimizer=optimizer,
                                          loss=criterion,
                                          metrics=[Accuracy()],
                                          backend="bigdl")

    orca_estimator.fit(data=train_iter, epochs=2, validation_data=test_iter,
                       checkpoint_trigger=EveryEpoch())

    res = orca_estimator.evaluate(data=test_iter)
    print("Accuracy of the network on the test images: %s" % res)
elif args.backend == "torch_distributed":
    orca_estimator = Estimator.from_torch(model=model_creator,
                                          optimizer=optim_creator,
                                          loss=nn.CrossEntropyLoss(),
                                          metrics=[Accuracy()],
                                          backend="torch_distributed",
                                          config={"lr": 2e-5})

    orca_estimator.fit(data=train_loader_creator, epochs=2, batch_size=batch_size,)

    res = orca_estimator.evaluate(data=test_loader_creator)
    for r in res:
        print(r, ":", res[r])
else:
    raise NotImplementedError("Only bigdl and torch_distributed are supported as the backend,"
                              " but got {}".format(args.backend))

stop_orca_context()

# start testing
print("***Finish training, start testing***")
model = model_creator({})
test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."


test_sen_set = [test_sen1, test_sen2]

for test_sen in test_sen_set:
    print(test_sen)
    test_sen = TEXT.preprocess(test_sen)
    test_sen = [[TEXT.vocab.stoi[x] for x in test_sen]]
    test_sen = np.asarray(test_sen)
    test_sen = torch.LongTensor(test_sen)
    test_tensor = Variable(test_sen, volatile=True)
    model.eval()
    output = model(test_tensor, 1)
    out = F.softmax(output, 1)
    if (torch.argmax(out[0]) == 1):
        print ("Sentiment: Positive")
    else:
        print ("Sentiment: Negative")
