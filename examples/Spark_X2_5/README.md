# XHToken/Spark-X2.5 模型部署说明

模型地址：[XHToken/Spark-X2.5-1.7B](https://modelscope.cn/models/XHToken/Spark-X2.5-1.7B) / [XHToken/Spark-X2.5-4B](https://modelscope.cn/models/XHToken/Spark-X2.5-4B)

Spark-X2.5 为混合注意力架构（full_attention + sliding_attention，sliding_window=512），
支持 1.7B 与 4B 两种规格，导出/转换命令通过 `--model_size` 区分。

## 1. 安装依赖库

安装本目录 `python/requirements.txt` 中的依赖（pytorch/transformers/onnx 等），用于导出 onnx：

```shell
pip install -r requirements.txt
```

## 2. 导出 onnx 模型

```shell
cd python

# 1.7B（默认）
python export_llm.py --modelscope

# 4B
python export_llm.py --modelscope --model_size 4B
```

导出产物在 `model/llm/` 目录（onnx、config.pkl、tokenizer.gguf、embed.bin）。

## 3. 转换 rknn 模型

```shell
cd python

# 1.7B（默认）
python export_rknn.py

# 4B
python export_rknn.py --model_size 4B
```

可通过 `--platform` 指定模型目标平台（默认 `rk1820`，可选 `rk1820`/`rk1828`/`rk3572`）：

```shell
python export_rknn.py --platform rk1820
```

> 说明：`rk1820`/`rk1828` 模型部署在 rk3588 板上以协处理器模式运行，因此第 4 节
> 板端编译使用 `-t rk3588`；若模型转换为 `rk3572`，则部署在 rk3572 板上，
> 板端编译对应改为 `./build-linux.sh -t rk3572 -a aarch64 -d Spark_X2_5`。

## 4. C++ 板端部署

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译（rk1820/rk1828 模型在 rk3588 板上以协处理器模式运行）
./build-linux.sh -t rk3588 -a aarch64 -d Spark_X2_5
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_spark_x2_5_demo/` 目录：

```
rknn_spark_x2_5_demo/
├── lib
│   └── librknn3_api.so
├── model
│   ├── Spark-X2.5-1.7B.embed.bin
│   ├── Spark-X2.5-1.7B.rknn
│   ├── Spark-X2.5-1.7B.safetensors
│   ├── Spark-X2.5-1.7B.tokenizer.gguf
│   └── Spark-X2.5-1.7B.weight
└── rknn_spark_x2_5_demo
```

### 4.1 部署到开发板

```bash
# 推送 demo 目录
adb push install/rk3588_linux_aarch64/rknn_spark_x2_5_demo/ /data
```

### 4.2 运行示例

```bash
adb shell
cd /data/rknn_spark_x2_5_demo

# 非思考模式（默认）
./rknn_spark_x2_5_demo model/Spark-X2.5-1.7B.rknn model/Spark-X2.5-1.7B.weight 0xff \
    model/Spark-X2.5-1.7B.tokenizer.gguf model/Spark-X2.5-1.7B.embed.bin 4096 128 \
    model/Spark-X2.5-1.7B.safetensors "你好"

# 思考模式：追加 --thinking
./rknn_spark_x2_5_demo model/Spark-X2.5-1.7B.rknn model/Spark-X2.5-1.7B.weight 0xff \
    model/Spark-X2.5-1.7B.tokenizer.gguf model/Spark-X2.5-1.7B.embed.bin 4096 128 \
    model/Spark-X2.5-1.7B.safetensors "你好" --thinking
```

参数说明：

| 参数 | 说明 |
| ---- | ---- |
| llm_model_path | rknn 模型路径 |
| llm_weight_path | 权重文件路径 |
| llm_core_mask | NPU 核掩码，`0xff` 为全部核心 |
| tokenizer_path | tokenizer.gguf 路径 |
| embedding_path | embed.bin 路径 |
| max_context_len | 最大上下文长度（需不超过转换时 `--max_context_len`） |
| max_new_tokens | 最大生成 token 数 |
| safetensors_path | rope_caches.safetensors 路径（RoPE 查表缓存，`position_embeddings_host_storage=True` 时必需） |
| prompt | 用户输入（可选） |
| --thinking | 开启思考模式（可选，默认非思考） |
