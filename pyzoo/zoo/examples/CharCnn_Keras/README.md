# Character Level CNNs in Keras

This repository contains Keras implementations for two Character-level Convolutional Neural Networks for text classification on AG's News Topic Classification Dataset.

The following models have been implemented:
 1. Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). NIPS 2015
 2. Yoon Kim, Yacine Jernite, David Sontag, Alexander M. Rush. [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615). AAAI 2016

Kim's CharCNN was originally part of an end-to-end trained pipeline for language modelling, but has been adapted for text classification.

## Usage

1. Install dependencies:

```
$ pip install -r requirements.txt
```

2. Specify the training and testing data sources and model hyperparameters in the `config.json` file.

3. Run the main.py file as below:

```sh
$ python main.py --model [model_name]
```

Replace `[model_name]` with either `zhang` or `kim` to run the desired model.

## Results

Coming soon.
