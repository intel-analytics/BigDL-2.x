# How to Use Scripts
## Set Environment Variables

Before using scripts, two environment variables should be set.

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the dist directory of Analytics Zoo
```
The ```dist``` directory can be found in following hierarchy after you build Analytics Zoo.

```
analytics-zoo 
 |---apps                   (directory)
 |---backend                (directory)
 |---dist                  *(directory)*
       |---bin              (directory)
       |---conf             (directory)
       |---lib              (directory)
       |---extra-resources  (directory)
 |---docs                   (directory)
 |---pyzoo                  (directory)
 |---scripts                (directory)
 |---zoo                    (directory)
 |---LICENSE                (file)
 |---README.md              (file)
 |---make-dist.sh           (file)
 |---pom.xml                (file)
```

## Run Scripts
After setting the necessary environment variables above, you can run those scripts. One example is shown as following.
```bash
./spark-submit-with-zoo.sh \
    --master your_master_of_spark \
    --driver-cores cores_number_of_driver  \
    --driver-memory memory_size_of_driver  \
    --total-executor-cores total_cores_number_of_executor  \
    --executor-cores cores_number_of_executor  \
    --executor-memory memory_size_of_executor \
    path_to_your_python_script_of_model
```
Note that not all the parameters are required. You only need to set the necessary ones.

* For YARN cluster:

  You can run Analytics Zoo programs on YARN clusters without changes to the cluster (e.g., no need to pre-install the Python dependencies). You can first package all the required Python dependencies into a virtual environment on the local node (where you will run the spark-submit command), and then directly use `spark-submit-with-zoo.sh` to run the Zoo Python program on the YARN cluster (using that virtual environment). 
     
   Please follow the steps below: 
     
   1. Make sure you already install such libraries(python-setuptools, python-dev, gcc, make, zip, pip) for creating virtual environment. If not, please install them first. For example, on Ubuntu, run these commands to install:
    ```
            apt-get update
            apt-get install -y python-setuptools python-dev
            apt-get install -y gcc make
            apt-get install -y zip
            easy_install pip
    ```	
   2. Create dependency virtualenv package
        - Run ```python_package.sh``` to create dependency virtual environment according to dependency descriptions in requirements.txt. You can add your own dependencies in requirements.txt. The current requirements.txt only contains dependencies for Analytics Zoo python examples and models.
        - After running this script, there will be venv.zip and venv directory generated in current directory. Use them to submit your python jobs.            
   3. Submit job with virtualenv package
        - YARN cluster mode.
            
        Before submit job, set the environment first:
        ```
                export PYSPARK_PYTHON=./venv.zip/venv/bin/python
                export VENV_HOME=your virtual environment directory
        ```
        Then run ```spark-submit-with-zoo.sh``` as below:
        ```
                spark-submit-with-zoo.sh \
                --master yarn-cluster \
                --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./venv.zip/venv/bin/python \
                --archives ${VENV_HOME}/venv.zip \
                --executor-memory 10g \
                --driver-memory 10g \
                --executor-cores 8 \
                --num-executors 2 \
                your python script and parameters
        ```
        - YARN client mode.
                        
        Before submit job, set the environment first:
        ```
                export VENV_HOME=your virtual environment directory
                export PYSPARK_DRIVER_PYTHON=${VENV_HOME}/venv/bin/python
                export PYSPARK_PYTHON=./venv.zip/venv/bin/python
        ```
        Then run ```spark-submit-with-zoo.sh``` as below:
        ```
                spark-submit-with-zoo.sh \
                --master yarn \
                --deploy-mode client \
                --executor-memory 10g \
                --driver-memory 10g \
                --executor-cores 16 \
                --num-executors 2 \
                --archives ${VENV_HOME}/venv.zip \
                your python script and parameters
        ```    
           
        __FAQ__
        
        In case you encounter the following errors when you create the environment package using the above command:
        
        1. virtualenv ImportError: No module named urllib3
            - Using python in anaconda to create virtualenv may cause this problem. Try using python default in your system instead of installing virtualenv in anaconda.
        2. AttributeError: 'module' object has no attribute 'sslwrap'
            - Try upgrading `gevent` with `pip install --upgrade gevent`.
