# Analytics Zoo Text Classification API

Analytics Zoo provides a convenient API to create a model used for text classification with different encoders. This model could be fed into NNFrames and BigDL Optimizer directly for training.

### Scala API
To create a `TextClassifier` model in Scala:
```scala
TextClassifier(classNum, tokenLength, sequenceLength = 500, encoder = "cnn", encoderOutputDim = 256)
```

Parameters:

* `classNum`: The number of text categories to be classified. Positive integer.
* `tokenLength`: The size of each word vector. Positive integer.
* `sequenceLength`: The length of a sequence. Positive integer. Default is 500.
* `encoder`: The encoder for input sequences. String. "cnn" or "lstm" or "gru" are supported. Default is "cnn".
* `encoderOutputDim`: The output dimension for the encoder. Positive integer. Default is 256.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/textclassification) for the Scala example that trains the `TextClassifier` model on 20 Newsgroup dataset and uses the model to do prediction.


### Python API
To create a `TextClassifier` model in Python:
```python
TextClassifier(class_num, token_length, sequence_length=500, encoder="cnn", encoder_output_dim=256)
```

Parameters:

* `class_num`: The number of text categories to be classified. Positive int.
* `token_length`: The size of each word vector. Positive int.
* `sequence_length`: The length of a sequence. Positive int. Default is 500.
* `encoder`: The encoder for input sequences. String. 'cnn' or 'lstm' or 'gru' are supported. Default is 'cnn'.
* `encoder_output_dim`: The output dimension for the encoder. Positive int. Default is 256.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/textclassification) for the Python example that trains the `TextClassifier` model on 20 Newsgroup dataset and uses the model to do prediction.
