We provide a built-in BERTClassifier in Analytics Zoo TFPark for Natural Language Processing (NLP) classification tasks based on [TFEstimator](../TFPark/estimator/) and BERT.

Bidirectional Encoder Representations from Transformers (BERT) is Google's state-of-the-art pre-trained NLP model.
You may refer to [here](https://github.com/google-research/bert) for more details.

BERTClassifier is a pre-built TFEstimator that takes the hidden state of the first token to do classification.

After constructing a BERTClassifier, you can directly call [train](../TFPark/estimator/#train), [evaluate](../TFPark/estimator/#evaluate) or [predict](../TFPark/estimator/#predict) 
in a distributed fashion.

```python
from zoo.tfpark.text.estimator import BERTClassifier

estimator = BERTClassifier(num_classes, bert_config_file, init_checkpoint=None, use_one_hot_embeddings=False, optimizer=None, model_dir=None)
```

* `num_classes`: Positive int. The number of classes to be classified.
* `bert_config_file`: The path to the json file for BERT configurations.
* `init_checkpoint`: The path to the initial checkpoint of the pre-trained BERT model if any. Default is None.
* `use_one_hot_embeddings`: Boolean. Whether to use one-hot for word embeddings. Default is False.
* `optimizer`: The optimizer used to train the estimator. It can either be an instance of 
tf.train.Optimizer or the corresponding string representation. Default is None if no training is involved.
* `model_dir`: The output directory for model checkpoints to be written if any. Default is None.
