# QA-CLIP ViT-L/14 导出 RKNN3 模型指南

模型地址：https://huggingface.co/TencentARC/QA-CLIP-ViT-L-14

## 模型导出命令

### LLM/Text 模型

```bash
cd examples/QAClip/python/llm
python export_llm.py
python export_rknn.py --onnx_path=chinese_clip_text.onnx --rknn_path=chinese_clip_text.rknn --platform=rk1828
```

说明：

- `export_llm.py` 默认从 `--model_path=../../models/QA-CLIP-ViT-L-14` 加载模型，导出 `./chinese_clip_text.onnx`，并自动生成运行时配套文件 `chinese_clip_text.config.pkl`、`chinese_clip_text.tokenizer.gguf`、`chinese_clip_text.embed.bin`。

导出后会生成 Text RKNN 模型、权重文件、embedding 文件和 tokenizer 文件：

- `chinese_clip_text.rknn`
- `chinese_clip_text.weight`
- `chinese_clip_text.embed.bin`
- `chinese_clip_text.tokenizer.gguf`

#### 多桶动态长度（Multi-bucket dynamic length）
- `export_llm.py` 将 seq 轴导出为动态轴（`dynamic_axes`），长度上限为 352。
- `export_rknn.py` 配置 3 个桶做 `dynamic_input`：`[1,128,768]` / `[1,256,768]` / `[1,352,768]`。
- 端测推理时按实际 token 数自动选择最小的可容纳桶，通过 `rknn3_set_shape` 切换到对应shape 再推理，可减少 padding 的无效计算。

### Vision 模型

```bash
cd examples/QAClip/python/vision
python export_vision.py
python export_rknn.py --onnx_path=chinese_clip_vision.onnx --rknn_path=chinese_clip_vision.rknn --dataset_path=../data/datasets.txt --platform=rk1828
```

说明：

- `export_vision.py` 默认从 `--model_path=../../models/QA-CLIP-ViT-L-14` 加载模型，导出 `./chinese_clip_vision.onnx`。
- `export_rknn.py` 的 `--dataset_path` 指向 `examples/QAClip/data/datasets.txt`（内容为 `demo.jpg`，量化用一张示例图）。

导出后会生成 Vision RKNN 模型和权重文件：

- `chinese_clip_vision.rknn`
- `chinese_clip_vision.weight`

## C Demo

### 构建

在仓库根目录执行：

```bash
./build-linux.sh -t rk3588 -a aarch64 -d QAClip
```

如需在其他平台运行，请将 `-t` 参数和导出 RKNN 时的 `--platform` 参数改为对应平台（`-t` 可选 `rk3588/rk3576/rk3572/x86`）。

### 运行

进入编译产物目录后执行：

```bash
./rknn_qaclipvit_demo chinese_clip_text.rknn chinese_clip_text.weight chinese_clip_text.embed.bin chinese_clip_text.tokenizer.gguf chinese_clip_vision.rknn chinese_clip_vision.weight demo.jpg 1
```

Demo 会分别计算图片与内置中文文本的相似度，并输出 softmax 后的概率。

## 所需模型文件

| 文件名 | 说明 |
|--------|------|
| chinese_clip_text.rknn | Text RKNN 模型 |
| chinese_clip_text.weight | Text 权重文件 |
| chinese_clip_text.embed.bin | Text embedding 文件 |
| chinese_clip_text.tokenizer.gguf | Text tokenizer 文件 |
| chinese_clip_vision.rknn | Vision RKNN 模型 |
| chinese_clip_vision.weight | Vision 权重文件 |
| demo.jpg | 测试图片 |
