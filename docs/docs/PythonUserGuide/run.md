You need to first [install](install.md) analytics-zoo, either [from pip](install.md/#install-from-pip) or [without pip](install.md/#install-without-pip).

**NOTE**: Only __Python 2.7__, __Python 3.5__ and __Python 3.6__ are supported for now.

---
## **Run after pip install**

***Use an Interactive Shell***
* Type `python` in the command line to start a REPL.
* Try to run the [example code](#example-code) to verify the installation.


***Use Jupyter Notebook***
* Start jupyter notebook as you normally do, e.g.

```bash
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

* Try to run the [example code](#example-code) to verify the installation.


***Configurations***

* Increase memory

```bash
export SPARK_DRIVER_MEMORY=20g
```

* Add extra jars or python packages

 &emsp; Set the environment variables `BIGDL_JARS` and `BIGDL_PACKAGES` __BEFORE__ creating `SparkContext`:
```bash
export BIGDL_JARS=...
export BIGDL_PACKAGES=...
```

---
## **Run without pip install**
- Note that __Python 3.6__ is only compatible with Spark 1.6.4, 2.0.3, 2.1.1, 2.2.0 and onwards. See [this issue](https://issues.apache.org/jira/browse/SPARK-19019) for more discussion.

***Set Environment Variables***
Set **SPARK_HOME** and **ANALYTICS_ZOO_HOME**:

* If you download Analytics Zoo from the [Release Page](../release-download.md):
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the path where you extract the analytics-zoo package
```

* If you build Analytics Zoo by yourself:
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the dist directory of Analytics Zoo
```

***Update spark-analytics-zoo.conf (Optional)***

If you have some customized properties in some files, which will be used with the `--properties-file` option
in `spark-submit/pyspark`, you can add these customized properties into ${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf.

---
***Run with pyspark***
```bash
${ANALYTICS_ZOO_HOME}/bin/pyspark-with-zoo.sh --master local[*]
```
* `--master` set the master URL to connect to
* `--jars` if there are extra jars needed.
* `--py-files` if there are extra python packages needed.

You can also specify other options available for pyspark in the above command if needed.

Try to run the [example code](#example-code) for verification.

---
***Run with spark-submit***
An Analytics Zoo Python program runs as a standard pyspark program, which requires all Python dependencies
(e.g., numpy) used by the program to be installed on each node in the Spark cluster. You can try
running the Analytics Zoo [TextClassification Python example](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/textclassification)
as follows:

```bash
${ANALTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh --master local[*] text_classification.py
```

---
***Run with Jupyter Notebook***

With the full Python API support in Analytics Zoo, users can use our package together with powerful notebooks
(such as Jupyter Notebook) in a distributed fashion across the cluster, combining Python libraries,
Spark SQL/dataframes and MLlib, deep learning models in Analytics Zoo, as well as interactive
visualization tools.

__Prerequisites__: Install all the necessary libraries on the local node where you will run Jupyter, e.g., 

```bash
sudo apt install python
sudo apt install python-pip
sudo pip install numpy scipy pandas scikit-learn matplotlib seaborn wordcloud
```

Launch the Jupyter Notebook as follows:
```bash
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh --master local[*]
```
* `--master` set the master URL to connect to
* `--jars` if there are extra jars needed.
* `--py-files` if there are extra python packages needed.

You can also specify other options available for pyspark in the above command if needed.

After successfully launching Jupyter, you will be able to navigate to the notebook dashboard using
your browser. You can find the exact URL in the console output when you started Jupyter; by default,
the dashboard URL is http://your_node:8888/

Try to run the [example code](#example-code) for verification.

---
***Run with virtual environment on Yarn***
If you already created Analytics Zoo dependency virtual environment according to Yarn cluster guide [here](install.md/#install-without-pip),
you can run python programs using Analytics Zoo as following examples.

- Note: please set BigDL_HOME, SPARK_HOME environment. Set VENV_HOME to the parent directory of venv.zip and venv directory. Replace VERSION with your BigDL version, like 0.5.0. If you don't install BigDL from source, replace ${BigDL_HOME}/pyspark/bigdl/examples/lenet/lenet.py with your python program which is using BigDL.
* Yarn cluster mode
```
    BigDL_HOME=
    SPARK_HOME=
    PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
    PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH
    VENV_HOME=
    
    PYSPARK_PYTHON=./venv.zip/venv/bin/python ${SPARK_HOME}/bin/spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./venv.zip/venv/bin/python \
    --master yarn-cluster \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --py-files ${PYTHON_API_PATH} \
    --archives ${VENV_HOME}/venv.zip \
    --conf spark.driver.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
    --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
    ${BigDL_HOME}/pyspark/bigdl/examples/lenet/lenet.py
```

* Yarn client mode
```
    BigDL_HOME=
    SPARK_HOME=
    PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
    PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH
    VENV_HOME=
    
    PYSPARK_DRIVER_PYTHON=${VENV_HOME}/venv/bin/python PYSPARK_PYTHON=./venv.zip/venv/bin/python ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 16 \
    --num-executors 2 \
    --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --py-files ${PYTHON_API_PATH} \
    --archives ${VENV_HOME}/venv.zip \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
    ${BigDL_HOME}/pyspark/bigdl/examples/lenet/lenet.py
 ```

---
## **Example code**

To verify if Analytics Zoo can run successfully, run the following simple code:

```python
import zoo.version
from zoo.common.nncontext import *
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.keras.layers import *

# Get the current Analytics Zoo version
zoo.version.__version__
# Create or get a SparkContext. This will also init the BigDL engine.
sc = get_nncontext()
# Create a Sequential model containing a Dense layer.
model = Sequential()
model.add(Dense(8, input_shape=(10, )))
```