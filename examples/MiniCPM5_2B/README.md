# MiniCPM5-2B

## 模型说明

模型下载：[openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)。

## Python 环境

使用 Python 3.12，在仓库根目录安装本用例依赖：

```bash
pip install -r examples/MiniCPM5_2B/requirements.txt
```

## 导出 ONNX

从仓库根目录进入用例的 `python` 目录，以下导出命令均在该目录执行：

```bash
cd examples/MiniCPM5_2B/python
python export_onnx.py \
  --model_path openbmb/MiniCPM5-2B \
  --output_path ../model/onnx/MiniCPM5-2B.onnx
```

| 参数 | 必填 | 默认值 | 含义 |
|---|---|---|---|
| `--model_path` | 否 | `openbmb/MiniCPM5-2B` | 本地模型权重目录或 Hugging Face / ModelScope 模型 ID |
| `--output_path` | 是 | 无 | 输出 ONNX 文件路径，以 `.onnx` 结尾 |
| `--no-quant` | 否 | 不启用 | 关闭 GRQ 量化 |
| `--cali_dataset` | 否 | `../../../datasets/llm_quant.json` | GRQ 校准数据 JSON 文件路径 |
| `--modelscope` | 否 | 不启用 | 从 ModelScope 下载模型，默认使用 Hugging Face |
| `-h` / `--help` | 否 | 无 | 显示帮助信息 |

关闭 GRQ：

```bash
python export_onnx.py \
  --model_path openbmb/MiniCPM5-2B \
  --output_path ../model/normal/MiniCPM5-2B.onnx \
  --no-quant
```

## 导出 RKNN

```bash
python export_rknn.py \
  --onnx_path ../model/onnx/MiniCPM5-2B.onnx \
  --output_path ../model/w4a16/MiniCPM5-2B.rknn \
  --dtype w4a16 \
  --target rk1828
```

| 参数 | 必填 | 默认值 | 含义 |
|---|---|---|---|
| `--onnx_path` | 是 | 无 | 输入 ONNX 文件路径，同目录需有 ONNX 导出步骤生成的同名 `.config.pkl` |
| `--output_path` | 是 | 无 | 输出 RKNN 文件路径，以 `.rknn` 结尾 |
| `--dtype` | 否 | `w4a16` | 量化方式 |
| `--target` | 否 | `rk1828` | 目标芯片 |
| `-h` / `--help` | 否 | 无 | 显示帮助信息 |
