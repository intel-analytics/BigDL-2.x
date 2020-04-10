# Analytics Zoo Getting Started

This document provides quick reference information regarding installing the Analytics Zoo, running the applications, and developing your own applications using the Analytics Zoo. 

## 1. Try Analytics Zoo
Users can try Analytics Zoo with Docker and Google Colab environments without installing it in your local environment. For more information: 

- Check the [Docker User Guide](https://analytics-zoo.github.io/master/#DockerUserGuide/)
- Check the [Google Colab Guide page](https://analytics-zoo.github.io/master/#ProgrammingGuide/run-notebook-colab/)

## 2. Install Analytics Zoo
Analytics Zoo releases for installation are available for Python and Scala users. For more information: 
- Check the [Python User Guide](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) 
- Check the [Scala User Guide](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/) 

## 3. Run Analytics Zoo Applications
Analytics Zoo applications can run on remote or cloud resources, such as YARN, K8s clusters, Databricks, or Google Dataproc. 

- 3.1 Run on YARN

Check the [instructions](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-on-yarn-after-pip-install) for running on YARN with pip installation 

Check the [instructions](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-with-conda-environment-on-yarn) for running on YARN with conda installation
 
- 3.2 Run on K8s

Check the [instructions](https://analytics-zoo.github.io) for how to run Analytics Zoo applicaiton on K8s

- 3.3 Run on Databricks

Check the [instructions](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/PlatformGuide/AnalyticsZoo-on-Databricks.md) for how to run Analytics Zoo applicaiton on Databricks

- 3.4 Run on Google Dataproc

Check the [instructions](https://analytics-zoo.github.io/master/#ProgrammingGuide/run-on-dataproc/) for how to provision the Google Dataproc environment and runn Analytics Zoo applications 

## 4. Develop Analytics Zoo Applications
Analytics Zoo provides rich APIs, built-in models and solutions for various needs of application development. 

- 4.1 TensorFlow

Check the [TFPark APIs](https://analytics-zoo.github.io/master/#ProgrammingGuide/TFPark/tensorflow/) for how to build and evaluate TensorFlow models, and develop training & inference pipeline with the TFPark APIs. 

- 4.2 PyTorch

Pytorch users can user either: 

* [NNFrame APIs](https://analytics-zoo.github.io/master/#APIGuide/PipelineAPI/nnframes/) to build and train PyTorch models along with Spark Dataframes and Spark ML Pipeline, or 

* [Estimator APIs](https://analytics-zoo.github.io/master/#APIGuide/PipelineAPI/estimator/#estimator) to train and evaluate PyTorch models

- 4.3 BigDL

BigDL users can use either: 

* [NNFrame APIs](https://analytics-zoo.github.io/master/#APIGuide/PipelineAPI/nnframes/) to build deep learning models along with Spark Dataframes and Spark ML Pipeline, or 

* High level [Keras-style APIs](https://analytics-zoo.github.io/master/#KerasStyleAPIGuide/Optimization/training/) to build deep learning training pipeline

- 4.4 Cluster Serving

Analytics Zoo Cluster Serving is a real-time distributed serving solution that supports inferencing with deep learning models on large clusters. Cluster Serving has been available since the Analytics Zoo 0.7.0 release. 

Follow the [Cluster Serving Program Guide](https://analytics-zoo.github.io/master/#ClusterServingGuide/ProgrammingGuide/) to understand the Cluster Serving architecture, workflow, how to use and customize Cluster Serving to your needs.  The [Cluster Serving API Guide](https://analytics-zoo.github.io/master/#ClusterServingGuide/APIGuide/) explains the APIs in more detail. 

- 4.5 AutoML

AutoML framework has been supported in Analytics Zoo since 0.8.0 release. The AutoML framework includes components such as FeatureTransformer, Model, SearchEngine, and Pipeline. Check the [AutoML Overview](https://analytics-zoo.github.io/master/#ProgrammingGuide/AutoML/overview/) for a high level description of the AutoML framework.

Analytics Zoo is also providing a reference use case with AutoML framework - Time Series Forecasting. Please check out the details in the [Program Guide](https://analytics-zoo.github.io/master/#ProgrammingGuide/AutoML/forecasting/) and [API Guide](https://analytics-zoo.github.io/master/#APIGuide/AutoML/time-sequence-predictor/). 
