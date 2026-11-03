# Basic Dataset Evaluation Guide

[中文](README.md)

This guide uses `lmms-eval` to evaluate visual datasets through an RKNN3 server. `mmbench_en_dev_static` is used as the example task and does not require a Judge Key or Hugging Face login. Its first run downloads and caches MMBench automatically. Other upstream tasks can be selected through `TASK_NAME`; their scoring rules are defined by the corresponding task.

## 1. Install the Environment

Python 3.11 is recommended. Choose either Conda or uv.

### Conda

```bash
# Python 3.11 is recommended
conda create -n lmms_eval python=3.11 -y
conda activate lmms_eval

pip install -U pip
pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  'torch==2.5.1+cpu' 'torchvision==0.20.1+cpu'
pip install \
  'git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git@4eb7d01b9d16ba42b5c67b724c49f347a0c6bee5'
```

### uv

If `uv` is not installed, run the official installer once. Open a new terminal and confirm that `uv --version` is available before continuing.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Python 3.11 is recommended
uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install --torch-backend=cpu \
  'torch==2.5.1+cpu' 'torchvision==0.20.1+cpu' \
  'git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git@4eb7d01b9d16ba42b5c67b724c49f347a0c6bee5'
```

Keep the selected environment activated before running the test. No manual MMBench download or Hugging Face Token setup is needed.

## 2. Start the Server and Run the Test

The following server command uses `Qwen3-VL-4B` with a `384x384` vision model. The server image size must match the client `IMAGE_WIDTH` and `IMAGE_HEIGHT` values.

```bash
cd /userdata
./rkllm3-server \
  --alias Qwen3-VL-4B \
  -m /userdata/Qwen3-VL-model/Qwen3-VL-4B-llm.rknn \
  --model2 /userdata/Qwen3-VL-model/Qwen3-VL-4B-vision_384_384.rknn \
  --vocab /userdata/Qwen3-VL-model/Qwen3-VL-4B-llm.tokenizer.gguf \
  --embed /userdata/Qwen3-VL-model/Qwen3-VL-4B-llm.embed.bin \
  --host 0.0.0.0 --port 8080 -c 1024 --n_predict 128 \
  --repeat-penalty 1.1 --presence-penalty 0.0 --frequency-penalty 0.0 \
  --top-k 1 --top-p 1.0 --temp 0 \
  --img-width 384 --img-height 384 \
  --img-start '<|vision_start|>' --img-end '<|vision_end|>' \
  --img-content '<|image_pad|>' --qwenvl
```

In the `test` directory, run the following command with an activated Conda or uv environment:

```bash
cd <rknn3-model-zoo>/tools/MMBench_EVAL

TASK_NAME=mmbench_en_dev_static \
SERVER_IP=<board_network_IP> \
SERVER_PORT=8080 \
IMAGE_WIDTH=384 IMAGE_HEIGHT=384 \
bash run_data_test.sh
```

The first run needs access to Hugging Face and automatically downloads and caches MMBench. Later runs reuse the local cache.

For another upstream task, replace `TASK_NAME` with its lmms-eval task name. Set `MODEL_NAME` or `API_KEY` when the server alias or authentication differs.

## 3. Expected Result

The following full-run result was measured with `Qwen3-VL-4B`, `384x384`, and `mmbench_en_dev_static` over 4329 samples:

| Tasks | Filter | n-shot | Metric | Value | Stderr |
| --- | --- | ---: | --- | ---: | --- |
| mmbench_en_dev_static | none | 0 | gpt_eval_score | 84.02 | N/A |
| mmbench_en_dev_static | none | 0 | submission | N/A | N/A |

Results are written to `tools/MMBench_EVAL/results/<task_name>_<timestamp>/`, for example `tools/MMBench_EVAL/results/mmbench_en_dev_static_<timestamp>/`. The full score for this example is `84.02061855670104`, with `4329/4329` successful requests.
