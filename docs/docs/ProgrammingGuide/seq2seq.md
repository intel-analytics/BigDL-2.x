Analytics Zoo provides Seq2seq model which is a general-purpose encoder-decoder framework that can be used for Chatbot, Machine Translation and more.

**Highlights**

1. Easy-to-use models, could be fed into NNFrames or BigDL Optimizer for training.
2. Support SimpleRNN, LSTM and GRU.
3. Support transform encoder states before fed into decoder

---
## **Build a Seq2seq model**
You can call the following API in Scala and Python respectively to create a `Seq2seq`.

**Scala**
```scala
val encoder = RNNEncoder[Float](rnnType="lstm", numLayer=3, hiddenSize=3, embedding=Embedding[Float](10, inputSize))
val decoder = RNNDecoder[Float](rnnType="lstm", numLayer=3, hiddenSize=3, embedding=Embedding[Float](10, inputSize))
val bridge = Bridge[Float](bridgeType="dense", decoderHiddenSize=3)
val model = Seq2seq[Float](encoder, decoder, inputShape=SingleShape(List(-1)), outputShape=SingleShape(List(-1)), bridge)
```

* `rnnType`: currently support "simplernn | lstm | gru"
* `numLayer`: number of layers
* `hiddenSize`: hidden size
* `embedding`: embedding layer
* `bridgeType`: currently only support "dense | densenonlinear"
* `input_shape`: shape of encoder input
* `output_shape`: shape of decoder input

**Python**
```python
encoder = RNNEncoder.initialize(rnn_tpye="LSTM", nlayers=1, hidden_size=4)
decoder = RNNDecoder.initialize(rnn_tpye="LSTM", nlayers=1, hidden_size=4)
bridge = Bridge.initialize(bridge_type="dense", decoder_hidden_size=4)
seq2seq = Seq2seq(encoder, decoder, input_shape=[2, 4], output_shape=[2, 4], bridge)
```

* `rnn_type`: currently support "simplernn | lstm | gru"
* `nlayers`: number of layers
* `hidden_size`: hidden size
* `bridge_type`: currently only support "dense | densenonlinear"
* `input_shape`: shape of encoder input
* `output_shape`: shape of decoder input

---
## **Train a Seq2seq model**
After building the model, we can use BigDL Optimizer to train it (with validation) using RDD of [Sample](https://bigdl-project.github.io/master/#APIGuide/Data/#sample).

**Scala**
```scala
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.nn.{TimeDistributedMaskCriterion, ClassNLLCriterion}

val optimizer = Optimizer(
model,
trainSet,
TimeDistributedMaskCriterion(
  ClassNLLCriterion(paddingValue = padId),
  paddingValue = padId
),
batchSize = 128)

optimizer
  .setOptimMethod(new Adagrad(learningRate = 0.01, learningRateDecay = 0.001))
  .setEndWhen(Trigger.maxEpoch(20))
  .optimize()
```

**Python**
```python
from bigdl.optim.optimizer import *

optimizer = Optimizer(
    model=seq2seq,
    training_rdd=train_rdd,
    criterion=TimeDistributedMaskCriterion(ClassNLLCriterion()),
    end_trigger=MaxEpoch(20),
    batch_size=128,
    optim_method=Adagrad(learningrate=0.01, learningrate_decay=0.001))

optimizer.set_validation(
    batch_size=128,
    trigger=EveryEpoch())
```

---
## **Do prediction**

**Scala**
```scala
val result = model.infer(input, startSign, maxSeqLen, stopSign, buildOutput)
```

**Python**

Python API is under development.