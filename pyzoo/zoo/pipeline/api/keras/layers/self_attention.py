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


import numpy as np
import math

from bigdl.nn.layer import Sum

from zoo.pipeline.api.keras.engine import ZooKerasLayer
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.models import Model
import zoo.pipeline.api.autograd as auto

if sys.version >= '3':
    long = int
    unicode = str

def layer_norm(x, w, b, e=1e-5):
    sizes = x.get_output_shape()[1:]
    u = auto.mean(x, len(sizes), True)
    s = auto.mean(auto.square(x - u), len(sizes), True)
    y = (x - u) / auto.sqrt(s + e)
    y = y * w + b
    return y

def gelu(x):
    y = (auto.square(x) * x * 0.044715 + x) * (math.sqrt(2 / math.pi))
    y = Activation("tanh")(y) + 1.0
    y = x * 0.5 * y
    return y

def split_heads(x, n_head, k=False):
    sizes = x.get_output_shape()[1:]
    shape = list(sizes + (sizes[-1]/n_head,))
    shape[-2] = n_head
    r = Reshape(shape)(x)
    if k:
        f = Permute((2, 3, 1))(r)
    else:
        f = Permute((2, 1, 3))(r)
    return f

def merge_heads(x):
    p = auto.contiguous(Permute((2, 1, 3))(x))
    sizes = p.get_output_shape()[1:]
    merge_sizes = list((sizes[0], sizes[-1]*sizes[-2]))
    m = Reshape(merge_sizes)(p)
    return m

class TransformerLayer(ZooKerasLayer):
    """
    A self attention layer

    # Arguments
    nBlock: block number
    resid_drop: drop probability off projection
    attn_drop: drop probability of attention
    n_head: head number
    mask_attention: whether unidirectional or bidirectional
    embedding_layer: embedding layer
    """
    def __init__(self, n_block, resid_drop, attn_drop,
                 n_head, mask_attention, embedding_layer, input_shape, bigdl_type="float"):
        self.resid_drop = resid_drop
        self.attn_drop = attn_drop
        self.n_head = n_head
        self.mask_attention = mask_attention
        self.seq_len = input_shape[0]
        self.bigdl_type = bigdl_type
        if mask_attention:
            mask_value = np.tril(np.ones((self.seq_len, self.seq_len), dtype=bigdl_type))
            self.mask_value = auto.Constant(data=mask_value.reshape((1, 1,
                                                                     self.seq_len, self.seq_len)))

        input = Input(shape=list(input_shape))
        embedding = embedding_layer(input)
        hidden_size = embedding.get_output_shape()[-1]

        next_input = embedding

        for _ in range(n_block):
            output = self.block(next_input, hidden_size)
            next_input = output

        model = Model(input, next_input)
        self.value = model.value

    def block(self, x, size):
        g = auto.Parameter(shape=(1, size), init_weight=np.ones((1, size), dtype=self.bigdl_type))
        b = auto.Parameter(shape=(1, size), init_weight=np.zeros((1, size), dtype=self.bigdl_type))
        g2 = auto.Parameter(shape=(1, size), init_weight=np.ones((1, size), dtype=self.bigdl_type))
        b2 = auto.Parameter(shape=(1, size), init_weight=np.zeros((1, size), dtype=self.bigdl_type))

        a = self.multi_head_self_attention(x, size)
        n = self.layer_norm(x + a, w=g, b=b)
        m = self.mlp(n, size)
        h = self.layer_norm(n + m, w=g2, b=b2)
        return h

    def multi_head_self_attention(self, x, size):
        c = Convolution1D(size * 3, 1, "normal", (0.0, 0.02))(x)
        query = c.slice(2, 0, size)
        key = c.slice(2, size, size)
        value = c.slice(2, size*2, size)
        q = split_heads(query, self.n_head)
        k = split_heads(key, self.n_head, k=True)
        v = split_heads(value, self.n_head)
        a = self.attn(q, k, v, True)
        m = merge_heads(a)
        n = Convolution1D(size, 1, "normal", (0.0, 0.02))(m)
        d = Dropout(self.resid_drop)(n)
        return d

    def attn(self, q, k, v, scale=False):
        w = auto.mm(q, k)
        if scale:
            w = w / math.sqrt(v.get_output_shape()[-1])

        if self.mask_attention:
            w = w * self.mask_value + (self.mask_value * (-1.0) + 1.0) * (-1e9)

        w = Activation("softmax")(w)
        w = Dropout(self.attn_drop)(w)
        w = auto.mm(w, v)
        return w

    def mlp(self, x, size):
        h = Convolution1D(size*4, 1, init="normal", limits=(0.0, 0.02))(x)
        a = gelu(h)
        h2 = Convolution1D(size, 1, init="normal", limits=(0.0, 0.02))(a)
        y = Dropout(self.resid_drop)(h2)
        return y

    @classmethod
    def init_with_default_embedding(cls, vocab=40990, seq_len=77, n_block=12, resid_drop=0.1,
                                    attn_drop=0.1, n_head=12, hidden_size=768,
                                    embedding_drop=0.1, mask_attention=True):
        """
        vocab: vocabulary size of training data, default is 40990
        seq_len: max sequence length of training data, default is 77
        n_block: block number, default is 12
        resid_drop: drop probability of projection, default is 0.1
        attn_drop: drop probability of attention, default is 0.1
        n_head: head number, default is 12
        hidden_size: is also embedding size
        embedding_drop: drop probability of embedding layer, default is 0.1
        mask_attention: whether unidirectional or bidirectional, default is true(unidirectional)
        """
        from bigdl.nn.layer import Squeeze
        embedding = Sequential()

        embedding.add(Reshape([seq_len * 2], input_shape=(seq_len, 2)))\
            .add(Embedding(vocab, hidden_size, input_length=seq_len * 2))\
            .add(Dropout(embedding_drop))\
            .add(Reshape((seq_len, 2, hidden_size)))\
            .add(KerasLayerWrapper(Sum(dimension=3, squeeze=True)))
        # walk around for bug #1208, need remove this line after the bug fixed
        embedding.add(KerasLayerWrapper(Squeeze(dim=3)))

        return TransformerLayer(n_block, resid_drop, attn_drop, n_head, mask_attention,
                                embedding, input_shape=(seq_len, 2))

class BERT(ZooKerasLayer):
    """
    A self attention layer.
    Input is a List which consists of 4 ndarrays.
    1. Token id ndarray: shape [batch, seqLen] with the word token indices in the vocabulary
    2. Token type id ndarray: shape [batch, seqLen] with the token types in [0, 1].
       0 menas `sentence A` and 1 means a `sentence B` (see BERT paper for more details).
    3. Position id ndarray: shape [batch, seqLen] with positions in the sentence.
    4. Attention_mask ndarray: shape [batch, seqLen] with indices in [0, 1].
       It's a mask to be used if the input sequence length is smaller than seqLen in
       the current batch.
    Output is a list which output the states of BERT layer

    # Arguments
    n_block: block number
    n_head: head number
    intermediate_size: The size of the "intermediate" (i.e., feed-forward)
    hidden_drop: The dropout probabilitiy for all fully connected layers
    attn_drop: drop probability of attention
    output_all_block: whether output all blocks' output
    embedding_layer: embedding layer 
    """
    def __init__(self, n_block, n_head, intermediate_size, hidden_drop, attn_drop,
                 output_all_block, embedding_layer, input_shape, bigdl_type="float"):
        self.hidden_drop = hidden_drop
        self.attn_drop = attn_drop
        self.n_head = n_head
        self.intermediate_size = intermediate_size
        self.output_all_block = output_all_block
        self.bigdl_type = bigdl_type
        self.seq_len = input_shape[0][0]

        word_input = Input(shape=input_shape[0])
        token_type_input = Input(shape=input_shape[1])
        position_input = Input(shape=input_shape[2])
        attention_mask = Input(shape=input_shape[3])

        e = embedding_layer([word_input, token_type_input, position_input])
        self.hidden_size = e.get_output_shape()[-1]
        extended_attention_mask = (- attention_mask + 1.0) * -10000.0

        next_input = e
        model_output = [None] * n_block
        model_output[0] = self.block(next_input, self.hidden_size, extended_attention_mask)

        for _ in range(n_block-1):
            output = self.block(model_output[_], self.hidden_size, extended_attention_mask)
            model_output[_+1] = output

        if output_all_block:
            model = Model([word_input, token_type_input, position_input, attention_mask], model_output)
        else:
            model = Model([word_input, token_type_input, position_input, attention_mask], model_output[-1])
        self.value = model.value

    def block(self, x, size, attention_mask):
        g = auto.Parameter(shape=(1, size), init_weight=np.ones((1, size), dtype=self.bigdl_type))
        b = auto.Parameter(shape=(1, size), init_weight=np.zeros((1, size), dtype=self.bigdl_type))
        g2 = auto.Parameter(shape=(1, size), init_weight=np.ones((1, size), dtype=self.bigdl_type))
        b2 = auto.Parameter(shape=(1, size), init_weight=np.zeros((1, size), dtype=self.bigdl_type))

        a = self.multi_head_self_attention(x, attention_mask, size)
        n = layer_norm(x + a, w=g, b=b, e=1e-12)
        m = self.mlp(n, size)
        h = layer_norm(n + m, w=g2, b=b2, e=1e-12)
        return h

    def multi_head_self_attention(self, x, attention_mask, size):
        attn_head_size = size / self.n_head
        all_head_size = self.n_head * attn_head_size
        query = Dense(all_head_size)(x)
        key = Dense(all_head_size)(x)
        value = Dense(all_head_size)(x)
        q = split_heads(query, self.n_head)
        k = split_heads(key, self.n_head, k=True)
        v = split_heads(value, self.n_head)
        a = self.attn(q, k, v, attention_mask)
        m = merge_heads(a)
        n = Dense(size)(m)
        d = Dropout(self.hidden_drop)(n)
        return d

    def attn(self, q, k, v, attention_mask):
        w = auto.mm(q, k)
        w = w / math.sqrt(v.get_output_shape()[-1])

        w = w + attention_mask

        w = Activation("softmax")(w)
        w = Dropout(self.attn_drop)(w)
        w = auto.mm(w, v)
        return w

    def mlp(self, x, size):
        h = Dense(self.intermediate_size)(x)
        a = gelu(h)
        h2 = Dense(size)(a)
        y = Dropout(self.hidden_drop)(h2)
        return y

    @classmethod
    def init_with_default_embedding(cls, vocab=40990, hidden_size=768, n_block=12, n_head=12,
                                    seq_len=77, intermediate_size=3072, hidden_drop=0.1,
                                    attn_drop=0.1, output_all_block=False, bigdl_type="float"):
        """
        vocab: vocabulary size of training data, default is 40990
        hidden_size: size of the encoder layers, default is 768
        n_block: block number, default is 12
        n_head: head number, default is 12
        seq_len: max sequence length of training data, default is 77
        intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        hidden_drop: drop probability of full connected layers, default is 0.1
        attn_drop: drop probability of attention, default is 0.1
        output_all_block: whether output all blocks' output, default is False
        """
        word_input = Input(shape=(seq_len,))
        token_type_input = Input(shape=(seq_len,))
        position_input = Input(shape=(seq_len,))
        word_embedding = Embedding(vocab, hidden_size, input_length=seq_len)(word_input)
        position_embedding = Embedding(seq_len, hidden_size, input_length=seq_len)(position_input)
        token_type_embedding = Embedding(2, hidden_size, input_length=seq_len)(token_type_input)
        embedding = word_embedding + position_embedding + token_type_embedding

        w = auto.Parameter(shape=(1, hidden_size), init_weight=np.ones((1, hidden_size), dtype=bigdl_type))
        b = auto.Parameter(shape=(1, hidden_size), init_weight=np.zeros((1, hidden_size), dtype=bigdl_type))
        after_norm = layer_norm(embedding, w, b, 1e-12)
        h = Dropout(hidden_drop)(after_norm)

        embedding_layer = Model([word_input, token_type_input, position_input], h)
        shape = ((seq_len,), (seq_len,), (seq_len,), (1, 1, seq_len))

        return BERT(n_block, n_head, intermediate_size, hidden_drop, attn_drop, output_all_block,
                                embedding_layer, input_shape=shape)
