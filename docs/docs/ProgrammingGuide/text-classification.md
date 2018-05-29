Analytics Zoo provides a set of pre-defined models that can be used for classifying texts with different encoders.

**Highlights**

1. Easy-to-use models, could be fed into NNFrames and BigDL Optimizer for training.
2. The encoders we support include CNN, LSTM and GRU.

---
## Build a TextClassifier model

**Scala**
```scala
val textClassifier = TextClassifier(classNum, tokenLength, sequenceLength = 500, encoder = "cnn", encoderOutputDim = 256)
```

* `classNum`: The number of text categories to be classified. Positive integer.
* `tokenLength`: The size of each word vector. Positive integer.
* `sequenceLength`: The length of a sequence. Positive integer. Default is 500.
* `encoder`: The encoder for input sequences. String. "cnn" or "lstm" or "gru" are supported. Default is "cnn".
* `encoderOutputDim`: The output dimension for the encoder. Positive integer. Default is 256.

**Python**
```python
text_classifier = TextClassifier(class_num, token_length, sequence_length=500, encoder="cnn", encoder_output_dim=256)
```

* `class_num`: The number of text categories to be classified. Positive int.
* `token_length`: The size of each word vector. Positive int.
* `sequence_length`: The length of a sequence. Positive int. Default is 500.
* `encoder`: The encoder for input sequences. String. 'cnn' or 'lstm' or 'gru' are supported. Default is 'cnn'.
* `encoder_output_dim`: The output dimension for the encoder. Positive int. Default is 256.


---
## Train a TextClassifier model
After building the model, we can use BigDL Optimizer to train it:

**Scala**
```scala
val optimizer = Optimizer(
  model = textClassifier,
  sampleRDD = trainRDD,
  criterion = ClassNLLCriterion[Float](logProbAsInput = false),
  batchSize = 128
)

optimizer
  .setOptimMethod(new Adagrad(learningRate = 0.01, learningRateDecay = 0.001))
  .setValidation(Trigger.everyEpoch, valRDD, Array(new Top1Accuracy), 128)
  .setEndWhen(Trigger.maxEpoch(20))
  .optimize()
```

**Python**
```python
optimizer = Optimizer(
    model=text_classifier,
    training_rdd=train_rdd,
    criterion=ClassNLLCriterion(logProbAsInput=False),
    end_trigger=MaxEpoch(20),
    batch_size=128,
    optim_method=Adagrad(learningrate=0.01, learningrate_decay=0.001))
optimizer.set_validation(
    batch_size=128,
    val_rdd=val_rdd,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()])
```


---
## Use a trained TextClassifier model to do prediction
After training the model, we can use the model to predict probabilities or text class labels.

**Scala**
```scala
// Predict for probability distributions
val results = textClassifier.predict(rdd)
// Predict for class labels
val resultClasses = textClassifier.predictClass(rdd)
```

**Python**
```python
# Predict for probability distributions
results = text_classifier.predict(rdd)
# Predict for class labels
result_classes = text_classifier.predict_class(rdd)
```

## Examples
We provide an example to trains the `TextClassifier` model on 20 Newsgroup dataset and uses the model to do prediction.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/textclassification) for the Scala example.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/textclassification) for the Python example.
