# Serving using BigDL-LLM and FastChat

FastChat is an open platform for training, serving, and evaluating large language model based chatbots. You can find the detailed information at their [homepage](https://github.com/lm-sys/FastChat).

BigDL-LLM can be easily integrated into FastChat so that user can use `BigDL-LLM` as a serving backend in the deployment.

<details>
<summary>Table of contents</summary>

- [Install](#install)
- [Start the service](#start-the-service)
  - [Launch controller](#launch-controller)
  - [Launch model worker(s) and load models](#launch-model-workers-and-load-models)
    - [BigDL model worker (deprecated)](#bigdl-model-worker-deprecated)
    - [BigDL worker](#bigdl-llm-worker)
    - [BigDL vLLM model worker](#vllm-model-worker)
  - [Launch Gradio web server](#launch-gradio-web-server)
  - [Launch RESTful API server](#launch-restful-api-server)

</details>

## Install

You may install **`bigdl-llm`** with `FastChat` as follows:

```bash
pip install --pre --upgrade bigdl-llm[serving]

# Or
pip install --pre --upgrade bigdl-llm[all]
```

To add GPU support for FastChat, you may install **`bigdl-llm`** as follows:

```bash
pip install --pre --upgrade bigdl-llm[xpu, serving] -f https://developer.intel.com/ipex-whl-stable-xpu

```

## Start the service

### Launch controller

You need first run the fastchat controller

```bash
python3 -m fastchat.serve.controller
```

### Launch model worker(s) and load models

Using BigDL-LLM in FastChat does not impose any new limitations on model usage. Therefore, all Hugging Face Transformer models can be utilized in FastChat.

#### BigDL model worker (deprecated)
<details>
<summary>details</summary>

> Warning: This method has been deprecated, please change to use `BigDL-LLM` [worker](#bigdl-llm-worker) instead.

FastChat determines the Model adapter to use through path matching. Therefore, in order to load models using BigDL-LLM, you need to make some modifications to the model's name.

For instance, assuming you have downloaded the `llama-7b-hf` from [HuggingFace](https://huggingface.co/decapoda-research/llama-7b-hf).  Then, to use the `BigDL-LLM` as backend, you need to change name from `llama-7b-hf` to `bigdl-7b`.The key point here is that the model's path should include "bigdl" and **should not include paths matched by other model adapters**.

Then we will use `bigdl-7b` as model-path.

> note: This is caused by the priority of name matching list. The new added `BigDL-LLM` adapter is at the tail of the name-matching list so that it has the lowest priority. If model path contains other keywords like `vicuna` which matches to another adapter with higher priority, then the `BigDL-LLM` adapter will not work.

A special case is `ChatGLM` models. For these models, you do not need to do any changes after downloading the model and the `BigDL-LLM` backend will be used automatically.

Then we can run model workers

```bash
# On CPU
python3 -m bigdl.llm.serving.fastchat.model_worker --model-path PATH/TO/bigdl-7b --device cpu

# On GPU
python3 -m bigdl.llm.serving.fastchat.model_worker --model-path PATH/TO/bigdl-7b --device xpu
```

If you run successfully using `BigDL` backend, you can see the output in log like this:

```bash
INFO - Converting the current model to sym_int4 format......
```

> note: We currently only support int4 quantization for this method.
</details>

#### BigDL-LLM worker
To integrate BigDL-LLM with `FastChat` efficiently, we have provided a new model_worker implementation named `bigdl_worker.py`.

To run the `bigdl_worker` on CPU, using the following code:
```bash
source bigdl-llm-init -t

# Available low_bit format including sym_int4, sym_int8, bf16 etc.
python3 -m bigdl.llm.serving.fastchat.bigdl_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "sym_int4" --trust-remote-code --device "cpu"
```


For GPU example:
```bash
# Available low_bit format including sym_int4, sym_int8, fp16 etc.
python3 -m bigdl.llm.serving.fastcaht.bigdl_worker --model-path lmsys/vicuna-7b-v1.5 --low-bit "sym_int4" --trust-remote-code --device "xpu"
```

For a full list of accepted arguments, you can refer to the main method of the `bigdl_worker.py`

#### BigDL vLLM model worker

We also provide the `vllm_worker` which uses the [vLLM](https://github.com/intel-analytics/BigDL/tree/main/python/llm/example/CPU/vLLM-Serving) engine for better hardware utilization.

To run using the `vLLM_worker`,  we don't need to change model name, just simply uses the following command:

```bash
# On CPU
python3 -m bigdl.llm.serving.fastchat.vllm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --device cpu

# On GPU
python3 -m bigdl.llm.serving.fastchat.vllm_worker --model-path REPO_ID_OR_YOUR_MODEL_PATH --device xpu
```

### Launch Gradio web server

```bash
python3 -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI with BigDL-LLM as the backend. You can open your browser and chat with a model now.

### Launch RESTful API server

To start an OpenAI API server that provides compatible APIs using BigDL-LLM backend, you can launch the `openai_api_server` and follow this [doc](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md) to use it.

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```
