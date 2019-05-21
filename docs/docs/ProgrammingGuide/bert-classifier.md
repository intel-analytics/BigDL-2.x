Analytics Zoo provides a built-in BERTClassifier in TFPark for Natural Language Processing (NLP) classification tasks based on [TFEstimator](../APIGuide/TFPark/estimator/) and BERT.

Bidirectional Encoder Representations from Transformers (BERT) is Google's state-of-the-art pre-trained NLP model.
You may refer to [here](https://github.com/google-research/bert) for more details.

BERTClassifier is a pre-built TFEstimator that takes the hidden state of the first token to do classification.

In this page, we show the general steps how to train and evaluate an [BERTClassifier](../APIGuide/TFPark/bert-classifier/) in a distributed fashion and use this estimator for distributed inference.


---
## **BERTClassifier Construction**
You can easily construct an estimator for classification based on BERT using the following API.

```python
from zoo.tfpark.text.estimator import BERTClassifier

estimator = BERTClassifier(num_classes, bert_config_file, init_checkpoint, optimizer=tf.train.AdamOptimizer(learning_rate), model_dir="/tmp/bert")
```

