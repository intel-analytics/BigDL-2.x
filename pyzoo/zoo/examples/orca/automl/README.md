## AutoXgboost example
This example illustrates how to use autoxgboost to do classification and regression.

### Run steps
#### 1. Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

#### 2. Prepare dataset

For AutoXGBoostClassifier, download dataset from [here] (http://kt.ijs.si/elena_ikonomovska/data.html)
we will get file 'airline_14col.data.bz2', unzip it with

bzip2 -d airline_14col.data.bz2

we will get `airline_14col.data` for training

For AutoXGBoostRegressor, download dataset from [here] (https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/orca/automl/incd.csv)


#### 5. Run the AutoXGBoostClassifier example

data_path=... // training data path. Local file system is supported.

##### * Run after pip install

You can easily use the following commands to run this example:

```bash
python path/to/AutoXGBoostClassifier.py --path ${data_path}
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

##### * Run with prebuilt package

```bash
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master local[4] \
    --driver-memory 10g \
    --executor-memory 10g \
    path/to/AutoXGBoostClassifier.py --path ${data_path}
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.
