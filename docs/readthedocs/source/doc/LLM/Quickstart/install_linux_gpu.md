# Install BigDL-LLM on Linux with Intel GPU

This guide demonstrates how to install BigDL-LLM on Linux with Intel GPUs. It applies to Intel Data Center GPU Flex Series and Max Series, as well as Intel Arc Series GPU.

BigDL-LLM currently supports the Ubuntu 20.04 operating system and later, and supports PyTorch 2.0 and PyTorch 2.1 on Linux. This page demonstrates BigDL-LLM with PyTorch 2.1. Check the [Installation](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#linux) page for more details.


## Install Intel GPU Driver

### For Linux kernel 6.2

* Install arc driver
    ```bash
    sudo apt-get install -y gpg-agent wget
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
    sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
    ```

  > <img src="https://llm-assets.readthedocs.io/en/latest/_images/wget.png" width=100%; />

* Install drivers

    ```bash
    sudo apt-get update

    sudo apt-get -y install \
        gawk \
        dkms \
        linux-headers-$(uname -r) \
        libc6-dev
        
    sudo apt install intel-i915-dkms  intel-fw-gpu

    sudo apt-get install -y gawk libc6-dev udev\
        intel-opencl-icd intel-level-zero-gpu level-zero \
        intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
        libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
        libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
        mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo
    
    sudo reboot
    ```

  > <img src="https://llm-assets.readthedocs.io/en/latest/_images/i915.png" width=100%; />

  > <img src="https://llm-assets.readthedocs.io/en/latest/_images/gawk.png" width=100%; />


* Configure permissions
    ```bash
    sudo gpasswd -a ${USER} render
    newgrp render

    # Verify the device is working with i915 driver
    sudo apt-get install -y hwinfo
    hwinfo --display
    ```


## Setup Python Environment

Install the Miniconda as follows if you don't have conda installed on your machine:
  ```bash
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

  bash Miniconda3-latest-Linux-x86_64.sh

  source ~/.bashrc

  # Verify the installation
  conda --version
  # rm Miniconda3-latest-Linux-x86_64.sh # if you don't need this file any longer
  ```
  > <img src="https://llm-assets.readthedocs.io/en/latest/_images/install_conda.png" alt="image-20240221102252569" width=100%; />


## Install oneAPI 
  ```
  wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
  
  sudo apt update

  sudo apt install intel-basekit
  ```
  > <img src="https://llm-assets.readthedocs.io/en/latest/_images/oneapi.png" alt="image-20240221102252565" width=100%; />

  > <img src="https://llm-assets.readthedocs.io/en/latest/_images/basekit.png" alt="image-20240221102252565" width=100%; />


## Install `bigdl-llm`

* With the `llm` environment active, use `pip` to install `bigdl-llm` for GPU:
  ```
  conda create -n llm python=3.9
  conda activate llm

  pip install --pre --upgrade bigdl-llm[xpu] --extra-index-url https://developer.intel.com/ipex-whl-stable-xpu
  ```

  > <img src="https://llm-assets.readthedocs.io/en/latest/_images/create_conda_env.png" alt="image-20240221102252564" width=100%; />
  
  > <img src="https://llm-assets.readthedocs.io/en/latest/_images/create_conda_env.png" alt="image-20240221102252564" width=100%; />


* You can verify if bigdl-llm is successfully installed by simply importing a few classes from the library. For example, execute the following import command in the terminal:
  ```bash
  source /opt/intel/oneapi/setvars.sh

  python

  > from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
  ```

  > <img src="https://llm-assets.readthedocs.io/en/latest/_images/verify_bigdl_import.png" alt="image-20240221102252562" width=100%; />


## Runtime Configurations

To use GPU acceleration on Linux, several environment variables are required or recommended before running a GPU example.

* For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series, we recommend:
  ```bash
  # Configure oneAPI environment variables. Required step for APT or offline installed oneAPI.
  # Skip this step for PIP-installed oneAPI since the environment has already been configured in LD_LIBRARY_PATH.
  source /opt/intel/oneapi/setvars.sh

  # Recommended Environment Variables for optimal performance
  export USE_XETLA=OFF
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  ```


* For Intel Data Center GPU Max Series, we recommend:
  ```bash
  # Configure oneAPI environment variables. Required step for APT or offline installed oneAPI.
  # Skip this step for PIP-installed oneAPI since the environment has already been configured in LD_LIBRARY_PATH.
  source /opt/intel/oneapi/setvars.sh

  # Recommended Environment Variables for optimal performance
  export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  export ENABLE_SDP_FUSION=1
  ```
  Please note that `libtcmalloc.so` can be installed by ```conda install -c conda-forge -y gperftools=2.10```.


## A Quick Example

Now let's play with a real LLM. We'll be using the [phi-1.5](https://huggingface.co/microsoft/phi-1_5) model, a 1.3 billion parameter LLM for this demostration. Follow the steps below to setup and run the model, and observe how it responds to a prompt "What is AI?". 

* Step 1: Open the **Anaconda Prompt** and activate the Python environment `llm` you previously created: 
   ```bash
   conda activate llm
   ```
* Step 2: If you're running on iGPU, set some environment variables by running below commands:
  > For more details about runtime configurations, refer to [this guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#runtime-configuration): 
  ```bash
  # Skip this step for PIP-installed oneAPI since the environment has already been configured in LD_LIBRARY_PATH.
  source /opt/intel/oneapi/setvars.sh

  # Recommended Environment Variables for optimal performance
  export USE_XETLA=OFF
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  ```
* Step 3: Create a new file named `demo.py` and insert the code snippet below.
   ```python
   # Copy/Paste the contents to a new file demo.py
   import torch
   from bigdl.llm.transformers import AutoModelForCausalLM
   from transformers import AutoTokenizer, GenerationConfig
   generation_config = GenerationConfig(use_cache = True)
   
   tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
   # load Model using bigdl-llm and load it to GPU
   model = AutoModelForCausalLM.from_pretrained(
       "tiiuae/falcon-7b", load_in_4bit=True, cpu_embedding=True, trust_remote_code=True)
   model = model.to('xpu')

   # Format the prompt
   question = "What is AI?"
   prompt = " Question:{prompt}\n\n Answer:".format(prompt=question)
   # Generate predicted tokens
   with torch.inference_mode():
       input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
       # warm up one more time before the actual generation task for the first run, see details in `Tips & Troubleshooting`
       # output = model.generate(input_ids, do_sample=False, max_new_tokens=32, generation_config = generation_config)
       output = model.generate(input_ids, do_sample=False, max_new_tokens=32, generation_config = generation_config).cpu()
       output_str = tokenizer.decode(output[0], skip_special_tokens=True)
       print(output_str)
   ```
   > Note: when running LLMs on Intel iGPUs with limited memory size, we recommend setting `cpu_embedding=True` in the `from_pretrained` function.
   > This will allow the memory-intensive embedding layer to utilize the CPU instead of GPU.

* Step 5. Run `demo.py` within the activated Python environment using the following command:
  ```bash
  python demo.py
  ```
   
   ### Example output
  
   Example output on a system equipped with an 11th Gen Intel Core i7 CPU and Iris Xe Graphics iGPU:
   ```
   Question:What is AI?
   Answer: AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines.
   ```

## Tips & Troubleshooting

### Warmup for optimial performance on first run
When running LLMs on GPU for the first time, you might notice the performance is lower than expected, with delays up to several minutes before the first token is generated. This delay occurs because the GPU kernels require compilation and initialization, which varies across different GPU types. To achieve optimal and consistent performance, we recommend a one-time warm-up by running `model.generate(...)` an additional time before starting your actual generation tasks. If you're developing an application, you can incorporate this warmup step into start-up or loading routine to enhance the user experience.

