# Analytics Zoo Getting Started

This document provides quick reference information regarding installing Analytics Zoo, running the applications, and developing your own applications using Analytics Zoo. 

## 1. Try Analytics Zoo
Users can easily try Analytics Zoo with Docker or Google Colab without installing it. For more information: 

- Check the [Docker User Guide](https://analytics-zoo.github.io/master/#DockerUserGuide/)
- Check the [Google Colab Guide page](https://analytics-zoo.github.io/master/#ProgrammingGuide/run-notebook-colab/)

## 2. Install Analytics Zoo
Analytics Zoo installation methods are available for Python and Scala users. 

#### 2.1 Python
- Check the [Python User Guide](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) 

#### 2.2 Scala
- Check the [Scala User Guide](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/) 

## 3. Run Analytics Zoo Applications
Analytics Zoo applications can run on remote or cloud resources, such as YARN, K8s clusters, Databricks, or Google Dataproc. 

#### 3.1 Run on YARN

- Check the [instructions](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-on-yarn-after-pip-install) for how to run Analytics Zoo applications on YARN with pip installation. 
- Check the [instructions](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-with-conda-environment-on-yarn) for how to run Analytics Zoo applications on YARN with conda installation.
 
#### 3.2 Run on K8s

- Check the [instructions](https://analytics-zoo.github.io) for how to run Analytics Zoo applicaitons on K8s.

#### 3.3 Run on Databricks

- Check the [instructions](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/PlatformGuide/AnalyticsZoo-on-Databricks.md) for how to run Analytics Zoo applicaitons on Databricks.

#### 3.4 Run on Google Dataproc

- Check the [instructions](https://analytics-zoo.github.io/master/#ProgrammingGuide/run-on-dataproc/) for how to provision the Google Dataproc environment and run Analytics Zoo applications. 

## 4. Develop Analytics Zoo Applications
Analytics Zoo provides comprehensive support for for building end-to-end, integrated data analytics and AI applications. 

#### 4.1 TensorFlow

- Check the [TFPark APIs](https://analytics-zoo.github.io/master/#ProgrammingGuide/TFPark/tensorflow/) for how to build and evaluate TensorFlow models, and develop training & inference pipeline with the TFPark APIs. 

#### 4.2 PyTorch

Pytorch users can user either: 

- [NNFrame APIs](https://analytics-zoo.github.io/master/#APIGuide/PipelineAPI/nnframes/) to run Spark ML Pipeline and Dataframe with PyTorch support, or 
- [Estimator APIs](https://analytics-zoo.github.io/master/#APIGuide/PipelineAPI/estimator/#estimator) to train and evaluate distributed PyTorch on Spark.

#### 4.3 BigDL

BigDL users can use either: 

- [NNFrame APIs](https://analytics-zoo.github.io/master/#APIGuide/PipelineAPI/nnframes/) to run Spark ML Pipeline and Dataframe with BigDL support, or 
- [Keras-style APIs for BigDL](https://analytics-zoo.github.io/master/#KerasStyleAPIGuide/Optimization/training/) to build deep learning pipeline.

#### 4.4 Cluster Serving

Analytics Zoo Cluster Serving is a real-time distributed serving solution for deep learning (including TF, PyTorch, Caffe, BigDL and OpenVINO). Cluster Serving has been available since the Analytics Zoo 0.7.0 release. 

Follow the [Cluster Serving Program Guide](https://analytics-zoo.github.io/master/#ClusterServingGuide/ProgrammingGuide/) to understand the Cluster Serving architecture, workflow, how to use and customize Cluster Serving to your needs.  The [Cluster Serving API Guide](https://analytics-zoo.github.io/master/#ClusterServingGuide/APIGuide/) explains the APIs in more detail. 

#### 4.5 AutoML
Analytics Zoo provides scalable AutoML support for time series prediction (including automatic feature generation, model selection and hyper-parameter tuning). AutoML framework has been supported in Analytics Zoo since 0.8.0 release. Check the [AutoML Overview](https://analytics-zoo.github.io/master/#ProgrammingGuide/AutoML/overview/) for a high level description of the AutoML framework.

Analytics Zoo is also providing a reference use case with AutoML framework - Time Series Forecasting. Please check out the details in the [Program Guide](https://analytics-zoo.github.io/master/#ProgrammingGuide/AutoML/forecasting/) and [API Guide](https://analytics-zoo.github.io/master/#APIGuide/AutoML/time-sequence-predictor/). 
