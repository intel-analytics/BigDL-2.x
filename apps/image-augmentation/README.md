# Image Augmentation
This is a simple example of image augmentation using Analytics ZOO API. We use various ways to transform images to augment the dataset.

## Environment
* Python 2.7/3.5/3.6 (numpy 1.11.1)
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)

## Install or download Analytics Zoo
* Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.  

## Install OpenCV
* The example uses OpenCV library to save image. Please install it before run this example.

## Run after pip install
* You can easily use the following commands to run this example:
```
     export SPARK_DRIVER_MEMORY=1g
     jupyter notebook --notebook-dir=./ --ip=* --no-browser 
```

* See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install. 


## Run with prebuild package
* Run the following command for Spark local mode (MASTER=local[*]) or cluster mode:
```
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

```
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 1g  \
    --executor-memory 1g
```
* See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.