# High dimension time series forecasting with zouwu TCMFForecaster 

This example demonstrates how to use Analytics Zoo Zouwu TCMFForecaster to run distributed training 
and inference for high dimension time series forecasting task.


## Environment
1. We recommend conda to set up your environment. Note that conda environment is required to run on 
yarn, but not strictly necessary for running on local. 
    ```bash
    conda create -n zoo python=3.7
    conda activate zoo
    ```

2. If you want to enable TCMFForecaster distributed training, it requires pre-install pytorch and horovod. You can follow the [horovod document](https://github.com/horovod/horovod/blob/master/docs/install.rst) to install the horovod and pytorch with Gloo support.
And here are the commands that work on for us on ubuntu 16.04. The exact steps may vary from different machines.

    ```bash
    conda install -y pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
    conda install -y cmake==3.16.0 -c conda-forge
    conda install cxx-compiler==1.0 -c conda-forge
    conda install openmpi
    HOROVOD_WITH_PYTORCH=1; HOROVOD_WITH_GLOO=1; pip install --no-cache-dir horovod==0.19.1
    pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl[ray]
    ```

    If you don't need distributed training. You only need to install pytorch in your environment.

    ```bash
    pip install torch==1.4.0 torchvision==0.5.0
    ```

3. Download and install nightly build analytics zoo whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip)).
    ```bash
    pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl[ray]
    ```

4. Install other packages
    ```bash
    pip install scikit-learn==0.22
    pip install pandas==1.0
    pip install requests
    ```

## Prepare data
The example use the public real-world electricity datasets. You can download by running [download datasets script](https://github.com/rajatsen91/deepglo/blob/master/datasets/download-data.sh). Note that we only need electricity.npy.

If you only want to try with dummy data, you can use the "--use_dummy_data" option.

## Run on local after pip install
```
python run_electricity.py --cluster_mode local
```

## Run on yarn cluster for yarn-client mode after pip install
```
python run_electricity.py --cluster_mode yarn
```

## Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`. You can refer to OrcaContext documents [here](https://analytics-zoo.github.io/master/#Orca/context/) for details.
* `--num_workers` The number of workers to be used in the cluster. You can change it depending on your own cluster setting. Default to be 2.
* `--cores` "The number of cpu cores you want to use on each node. Default to be 4.
* `--memory` The memory you want to use on each node. Default to be 10g
* `--data_dir` The directory of electricity data file.
* `--use_dummy_data` Whether to use dummy data. Default to be False.
* `--smoke` Whether to run smoke test. Smoke test run 1 iteration for each stage and run 2 iterations alternative training. Default to be False.
* `--predict_local` You should enable predict_local if want to run distributed training on yarn and run distributed inference on local."
* `--num_predict_cores` The number of cores you want to use for prediction on local. You should only parse this arg if you have set predict_local to true.
* `--num_predict_workers` The number of workers you want to use for prediction on local. You should only parse this arg if you have set predict_local to true.