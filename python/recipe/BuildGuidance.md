# Build Conda Package Guidance

## Check your envs
Because of IPEX we use are build on python3.7 only, so we are supposed to use [Anaconda based on python37](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2019.03-Linux-x86_64.sh) to build this package, please check if your base envs is on python3.7.
Install conda-build:
```bash
conda install conda-build
```
## Conda Recipe
Details at https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html

## Build
For example, we plan to build a package based on python3.7
1. Just run:
    ```bash
    cd orca-lite-poc

    conda build . -c chriskafka -c pytorch -c conda-forge --python 3.7
    ```
   What's more, we have pointed out building a `noarch:python` conda package in meta.yaml, so it's ok if you are not specfying --python. It will output a package ends with `py_0` indicated its cross-platform.  
1. Output on success:
    ```
    ####################################################################################
    Resource usage summary:

    Total time: 0:06:57.2
    CPU usage: sys=0:00:01.4, user=0:00:04.6
    Maximum memory usage observed: 181.3M
    Total disk usage observed (not including envs): 3.2K


    ####################################################################################
    Source and build intermediates have been left in /usr/local/conda-bld.
    There are currently 7 accumulated.
    To remove them, you can run the ```conda build purge``` command
    ```
    And you will see `/usr/local/conda-bld/linux-64/orca_lite-1.2.0-py37_0.tar.bz2` exist. But installing tar file directly does not resolve dependencies, refer to  https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages.

1. Upload conda package:
   
   Register a Anaconda account at https://anaconda.org/. Install and login Anaconda locally:
   ```
   conda install anaconda-client
   anaconda login
   ```
   Then you can upload your package by:
   ```
   anaconda upload /opt/conda/conda-bld/linux-64/orca_lite-0.0.2-py39_0.tar.bz2
   ```
   Now you can see a orca_lite package lies on your anaconda dashbroad.

1. Test your conda package:
    ```bash
    conda create -n test37 python=3.7
    conda activate test37

    conda install orca_lite -c conda-forge -c pytorch -c chriskafka
    ```
    Note that default channel may conflict with pytorch channel when it comes to torch, so we specify channels here.
    ```bash
    python -c "import orca_lite"
    conda list | grep torch
    pip list | grep ipex
    ```
    Check if versions are set right.

References:

1.https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html [recommended]
1.https://docs.anaconda.com/anacondaorg/user-guide/getting-started/
