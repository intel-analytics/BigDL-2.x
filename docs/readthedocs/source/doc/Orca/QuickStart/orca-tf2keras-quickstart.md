# TensorFlow 2 Quickstart

---

<a target="_blank" href="https://colab.research.google.com/github/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/tf2_keras_lenet_mnist.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>&nbsp; <a target="_blank" href="https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/tf2_keras_lenet_mnist.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>

---

**In this guide we will describe how to to scale out _TensorFlow 2_ programs using Orca in 4 simple steps.** (_[TensorFlow 1.5](./orca-tf-quickstart.md) and [Keras 2.3](./orca-keras-quickstart.md) guides are also available._)

### **Step 0: Prepare Environment**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../../UserGuide/python.md) for more details.

```bash
conda create -n zoo python=3.7 # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo[ray] # install either version 0.9 or latest nightly build
pip install tensorflow==2.3.0
```

### **Step 1: Init Orca Context**
```python
if args.cluster_mode == "local":
    init_orca_context(cluster_mode="local", cores=4) # run in local mode
elif args.cluster_mode == "k8s":
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2) # run on K8s cluster
elif args.cluster_mode == "yarn":
    init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2) # run on Hadoop YARN cluster
```

This is the only place where you need to specify local or distributed mode. View [Orca Context](./../Overview/orca-context.md) for more details.

**Note:** You should `export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir` when running on Hadoop YARN cluster. View [Hadoop User Guide](./../../UserGuide/hadoop.md) for more details.

### **Step 2: Define the Model**

You can then define the Keras model in the _Creator Function_ using the standard TensroFlow 2 APIs.

```python
import tensorflow as tf

def model_creator(config):
    model = tf.keras.Sequential(
        [tf.keras.layers.Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                input_shape=(28, 28, 1), padding='valid'),
         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
         tf.keras.layers.Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                                padding='valid'),
         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(500, activation='tanh'),
         tf.keras.layers.Dense(10, activation='softmax'),
         ]
    )

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```
### **Step 3: Define Train Dataset**

You can define the dataset in the _Creator Function_ using standard [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) APIs. Orca also supports [Spark DataFrame](https://spark.apache.org/docs/latest/sql-programming-guide.html) and [Orca XShards](../Overview/data-parallel-processing.md).

```python
def preprocess(x, y):
    x = tf.cast(tf.reshape(x, (28, 28, 1)), dtype=tf.float32) / 255.0
    return x, y

def train_data_creator(config, batch_size):
    (train_feature, train_label), _ = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices((train_feature, train_label))
    dataset = dataset.repeat()
    dataset = dataset.map(preprocess)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    return dataset

def val_data_creator(config, batch_size):
    _, (val_feature, val_label) = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices((val_feature, val_label))
    dataset = dataset.repeat()
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    return dataset
```

### **Step 4: Fit with Orca Estimator**

First, create an Estimator.

```python
from zoo.orca.learn.tf2 import Estimator

est = Estimator.from_keras(model_creator=model_creator, workers_per_node=2)
```

Next, fit and evaluate using the Estimator. 
```python
batch_size = 320
stats = est.fit(train_data_creator,
                epochs=5,
                batch_size=batch_size,
                steps_per_epoch=60000 // batch_size,
                validation_data=val_data_creator,
                validation_steps=10000 // batch_size)
                
est.save("/tmp/mnist_keras.ckpt")

stats = est.evaluate(val_data_creator, num_steps=10000 // batch_size)
est.shutdown()
print(stats)
```

That's it, the same code can run seamlessly in your local laptop and the distribute K8s or Hadoop cluster.

**Note:** You should call `stop_orca_context()` when your program finishes.
