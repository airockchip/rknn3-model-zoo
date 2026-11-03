# 基础数据集测试说明

[English](README_EN.md)

本说明使用 `lmms-eval` 通过 RKNN3 Server 进行数据集测试。以下以 `mmbench_en_dev_static` 为例；该任务不需要 Judge Key 或 Hugging Face 登录。首次运行会自动下载并缓存 MMBench。其他任务可通过修改 `TASK_NAME` 运行，具体评分规则以对应任务定义为准。

## 1. 安装环境

建议使用 Python 3.11。Conda 与 uv 二选一即可。

### Conda

```bash
# 建议使用 Python 3.11
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

若系统尚未安装 `uv`，先执行一次官方安装命令；重新打开终端后确认 `uv --version` 可用。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# 建议使用 Python 3.11
uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install --torch-backend=cpu \
  'torch==2.5.1+cpu' 'torchvision==0.20.1+cpu' \
  'git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git@4eb7d01b9d16ba42b5c67b724c49f347a0c6bee5'
```

安装完成后保持对应环境处于激活状态，再执行下面的测试命令。无需手动下载 MMBench 数据集或配置 Hugging Face Token。

## 2. 启动 Server 并运行测试

板端以 `Qwen3-VL-4B` 分辨率384*384为例启动服务。服务端图像尺寸与客户端的 `IMAGE_WIDTH`、`IMAGE_HEIGHT` 保持一致。

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

进入 `test` 目录，在已激活的 Conda 或 uv 环境中通过网络访问板端：

```bash
cd <rknn3-model-zoo>/tools/MMBench_EVAL

TASK_NAME=mmbench_en_dev_static \
SERVER_IP=<板端网络IP> \
SERVER_PORT=8080 \
IMAGE_WIDTH=384 IMAGE_HEIGHT=384 \
bash run_data_test.sh
```

首次运行需要能访问 Hugging Face，脚本会自动下载并缓存 MMBench；后续运行复用本机缓存。

测试其他上游任务时，将 `TASK_NAME` 替换为对应的 lmms-eval 任务名即可。服务别名或鉴权方式不同，可额外设置 `MODEL_NAME` 和 `API_KEY`。

## 3. 预期结果

使用 `Qwen3-VL-4B`、`384x384`、`mmbench_en_dev_static` 完成全量 4329 条测试后，实际结果如下：

| Tasks | Filter | n-shot | Metric | Value | Stderr |
| --- | --- | ---: | --- | ---: | --- |
| mmbench_en_dev_static | none | 0 | gpt_eval_score | 84.02 | N/A |
| mmbench_en_dev_static | none | 0 | submission | N/A | N/A |

结果目录为 `tools/MMBench_EVAL/results/<任务名>_<时间戳>/`，例如 `tools/MMBench_EVAL/results/mmbench_en_dev_static_<时间戳>/`。本示例的完整分数为 `84.02061855670104`，请求成功 `4329/4329`。
