# WeMM-Embedding-2B

## 模型说明

WeMM-Embedding-2B 是基于 Qwen3.5-2B 的多模态向量模型，支持文本、图片和视频输入，输出 2048 维向量。

模型地址：[tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B)

## Python 环境

```bash
pip install -r requirements.txt
```

RKNN 转换需要安装 `rknn3-toolkit`。

## 导出 ONNX

```bash
# LLM
python python/llm/export_onnx.py \
  --model_path tencent/WeMM-Embedding-2B \
  --output_path /path/to/work/wemm-embedding-2b-llm.onnx

# image
python python/vision/export_onnx.py \
  --model_path tencent/WeMM-Embedding-2B \
  --output_path /path/to/work/wemm-embedding-2b-vision-image.onnx \
  --mode image

# video
python python/vision/export_onnx.py \
  --model_path tencent/WeMM-Embedding-2B \
  --output_path /path/to/work/wemm-embedding-2b-vision-video.onnx \
  --mode video
```

LLM 导出同时生成 `.config.pkl`、`.tokenizer.gguf` 和 `.embed.bin`。

## 导出 RKNN

```bash
python python/llm/export_rknn.py \
  --onnx_path /path/to/work/wemm-embedding-2b-llm.onnx \
  --output_path /path/to/model/wemm-embedding-2b-llm.rknn \
  --target_platform rk1820 \
  --core_num 8 \
  --quantized_dtype w4a16 \
  --context_length 4096

python python/vision/export_rknn.py \
  --onnx_path /path/to/work/wemm-embedding-2b-vision-image.onnx \
  --output_path /path/to/model/wemm-embedding-2b-vision-image.rknn \
  --target_platform rk1820 \
  --core_num 8 \
  --quantized_dtype w4a16

python python/vision/export_rknn.py \
  --onnx_path /path/to/work/wemm-embedding-2b-vision-video.onnx \
  --output_path /path/to/model/wemm-embedding-2b-vision-video.rknn \
  --target_platform rk1820 \
  --core_num 8 \
  --quantized_dtype w4a16
```

## 编译

```bash
export GCC_COMPILER=/path/to/aarch64-none-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d WeMM-Embedding -b Release
```

编译结果位于 `install/rk3588_linux_aarch64/rknn_WeMM-Embedding_demo/`。

## 部署

```bash
adb push install/rk3588_linux_aarch64/rknn_WeMM-Embedding_demo/. \
  /userdata/rknn_wemm_embedding_demo/
adb push /path/to/model/. \
  /userdata/rknn_wemm_embedding_demo/model/
```

`model` 目录需包含：

```text
wemm-embedding-2b-llm.rknn
wemm-embedding-2b-llm.weight
wemm-embedding-2b-llm.tokenizer.gguf
wemm-embedding-2b-llm.embed.bin
wemm-embedding-2b-vision-image.rknn
wemm-embedding-2b-vision-image.weight
wemm-embedding-2b-vision-video.rknn
wemm-embedding-2b-vision-video.weight
```

## 板端推理

```bash
cd /userdata/rknn_wemm_embedding_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
```

文本：

```bash
./rknn_wemm_embedding_demo \
  --model_path ./model \
  --text "hello world" \
  --output_path ./text.npy
```

图片：

```bash
./rknn_wemm_embedding_demo \
  --model_path ./model \
  --image_path ./model/test_image.jpg \
  --text "Represent this image." \
  --output_path ./image.npy
```

视频：

```bash
./rknn_wemm_embedding_demo \
  --model_path ./model \
  --video_frames_path ./model/video_frames.txt \
  --video_fps 2 \
  --text "Represent this video." \
  --output_path ./video.npy
```

原始视频先在主机抽帧并推送到板端：

```bash
mkdir -p frames
ffmpeg -i input.mp4 -vf 'fps=2' -q:v 2 frames/%06d.jpg
find frames -name '*.jpg' | sort | sed 's#^#./model/#' > video_frames.txt
adb push frames /userdata/rknn_wemm_embedding_demo/model/
adb push video_frames.txt /userdata/rknn_wemm_embedding_demo/model/
```

## 测试资源来源

测试图片和视频帧来自 `VLM2Vec/MMEB-V3`：

- `model/test_image.jpg`：来自 `MCMR/images/41g+4WyEw6L._AC_.jpg`。
- `model/video_frames/`：来自 `video_mret/MomentSeeker/video_frames/videos_RoadAccidents014_x264_74.00_82.00/`。

MMEB-V3 仓库采用 MIT License，原始数据集遵循各自的许可条款。
