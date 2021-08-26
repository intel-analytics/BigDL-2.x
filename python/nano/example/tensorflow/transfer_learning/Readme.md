# BigDL-Nano Transfer Learning Example with Tensorflow

This example describes how BigDL Nano optimizes transfer learning. 
We use cat-vs-dog data sets to train a partial freezen neural network based on Mobile Net V2, and train unfreezen trainable variables through transfer learning.

This example is migrate from the tensorflow tutorial notebook at 
https://github.com/tensorflow/docs/blob/r2.4/site/en/tutorials/images/transfer_learning.ipynb

## Quick Start
1. Prepare Envrionment

    You can install the necessary packages with the following command
    ```
    pip install bigdl-nano[tf]
    ```
2. Run the Example

    You can run this example in your conda environment with the following command:
    ```
    python  transfer_learning.py
    ```


## Workflow and Results
We use the Mobile Net V2 as our base model, build our model combined with Preprocessing, Prediction and other Layers. The initial loss and accuary will be printed before training steps like this:

```
Number of trainable Varaiables: 2
26/26 [==============================] - 5s 87ms/step - loss: 0.9020 - accuracy: 0.4468
initial loss: 0.90
initial accuracy: 0.45
```

Then we freeze the base model and train other layers in the model. After 10 epoches of training, We unfreeze the last 100 layers of base model and continued to train on the basis of previous training. We will get the final accuracy after the evaluation of the model at the end of workflow.

```
Number of trainable variaables now: 56
Epoch 10/20
63/63 [==============================] - 14s 172ms/step - loss: 0.1472 - accuracy: 0.9390 - val_loss: 0.0623 - val_accuracy: 0.9790
Epoch 11/20
63/63 [==============================] - 10s 149ms/step - loss: 0.1188 - accuracy: 0.9480 - val_loss: 0.0571 - val_accuracy: 0.9802
...
```

After the training, we will evaluate the model. You can check the accuracy at the end of workflow. 

```
6/6 [==============================] - 1s 66ms/step - loss: 0.0351 - accuracy: 0.9844
Test accuracy : 0.984375
```