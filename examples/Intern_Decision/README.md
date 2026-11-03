# Intern-Decision

## 模型说明

Intern-Decision 支持文本和图文决策，可用于分类、是非判断和评分任务。

本用例以 Intern-Decision-0.8B 完成验证；其他规格尚未验证。以下操作与结果以 Intern-Decision-0.8B 为例。

模型原始权重下载链接：[Intern-Decision-0.8B（Hugging Face）](https://huggingface.co/internlm/Intern-Decision-0.8B)。

## 模型接口

Demo 提供以下两种用例：

| 用例 | 输入 | 输出 |
|---|---|---|
| 客服分流 | 用户消息文本 | 账号／支付／其他类别及概率、紧急程度评分 |
| 街景判断 | 街景图片 | 大型车辆类别及概率、是否有人及肯定概率 |

## 推理流程

```text
用户文本 + 任务配置                       图片（仅图文任务）
        │                                        │
        ▼                                        ▼
CPU 构建 prompt                          CPU 缩放图片
选项映射为 A/B/C/...                     uint8 RGB NHWC
        │                                        │
        ▼                                        ▼
分词、查表得到文本 embedding              Vision RKNN
        │                                        │
        │                                        ▼
        │                                    图像特征
        │                                        │
        └──────────────────┬─────────────────────┘
                           ▼
                       LLM RKNN
             每个决策字段独立 session、一次 prefill
             输入截至该字段 <decision> 之前的前缀
                           │
                           ▼
                 输出 62 个候选符号的 logits
                           │
                           ▼
              CPU 取当前字段有效选项的 logits
                softmax + 温度校准得到选项概率
                           │
                           ▼
               分类 / 是非判断 / 概率加权评分
                           │
                           ▼
                 终端展示，可选保存为 JSON
```

候选符号为 `A–Z`、`a–z`、`0–9`，具体含义由任务配置中的选项决定。多个决策字段分别推理，复用同一次图像编码得到的特征；纯文本任务跳过视觉分支。

## Python 环境

在 `examples/Intern_Decision` 目录下，使用已安装 RKNN3 Toolkit 的 Python 环境安装依赖。模型导出也在此目录执行。

```bash
pip install -r requirements.txt
```

## 导出 ONNX

```bash
python python/export_onnx.py \
  --model_path internlm/Intern-Decision-0.8B \
  --output_path model/onnx --stage all
```

- `--model_path`：本地权重目录或远端模型 ID。远端默认从 Hugging Face 下载；添加 `--modelscope` 后，视觉和文本权重均从 ModelScope 下载，本地目录不下载。
- `--output_path`：ONNX 输出目录。
- `--stage`：`vision` 导出视觉模型，`llm` 导出文本模型，`all` 导出两者。

文件名使用远端模型 ID 的最后一段或本地权重目录名，例如 `Intern-Decision-0.8B-vision.onnx`、`Intern-Decision-0.8B-llm.onnx`；配套文件沿用对应前缀。本地权重目录请使用完整模型名。**如果有特殊需要，可自行修改代码。**

## 导出 RKNN

```bash
python python/export_rknn.py \
  --onnx_path model/onnx \
  --output_path model/deploy --stage all --platform rk1820
```

- `--onnx_path`：ONNX 模型目录。
- `--output_path`：RKNN 输出目录，不能与 ONNX 目录相同。
- `--stage`：`vision` 转换视觉模型，`llm` 转换文本模型，`all` 转换两者，默认 `all`。
- `--platform`：目标芯片，可选 `rk1820`、`rk1828`、`rk3572`，默认 `rk1820`；前两者使用 8 核，RK3572 使用 1 核。

输入文件命名为 `<模型名>-vision.onnx` 和 `<模型名>-llm.onnx`，同一目录只放一套模型；输出文件保留对应前缀。

## 编译

返回仓库根目录，设置交叉编译器前缀并编译：

```bash
cd ../..
export GCC_COMPILER=/path/to/toolchain/bin/aarch64-none-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d Intern_Decision -b Release
```

RK3572 板端使用 `-t rk3572` 编译，推送时使用对应的 `rk3572_linux_aarch64` 目录。

将安装目录推送到板端：

```bash
adb push install/rk3588_linux_aarch64/rknn_Intern_Decision_demo /userdata/
```

## 板端运行

```bash
cd /userdata/rknn_Intern_Decision_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

# 客服分流：更换文本即可判断新的请求
./rknn_intern_decision_demo --model_path model --task_path model/customer_service.json \
  --text "我无法登录账号，请帮我恢复访问。"

./rknn_intern_decision_demo --model_path model --task_path model/customer_service.json \
  --text "我被重复扣款了，请帮我处理。"

# 街景判断：更换图片路径即可
./rknn_intern_decision_demo --model_path model --task_path model/street_scene.json \
  --image_path model/bus.jpg
```

| 参数 | 说明 |
|---|---|
| `--model_path` | 模型目录，必填。 |
| `--task_path` | 任务 JSON 文件路径，必填，可与模型位于不同目录。 |
| `--text` | 本次请求的文本。 |
| `--image_path` | 本次请求的图片路径。 |
| `--temperature` | 可选，概率校准温度，必须为有限正数。默认采用 0.8B 的 `2.747760550703`（以 float 存储）；使用其他规格时请指定对应温度。 |
| `--output_path` | 可选，保存包含完整概率和 logits 的 JSON，例如 `--output_path result.json`。 |

- `--text` 和 `--image_path` 至少提供一个，也可同时提供；只传图片时使用默认文本“请根据图片回答问题。”。
- 所有相对路径均相对于当前工作目录。
- 结果直接显示在终端。

模型文件按后缀扫描（仅目录第一层），可修改文件名前缀：

- `.tokenizer.gguf` 和 `.embed.bin` 各一个；与 tokenizer 同名前缀的 `.rknn` 是文本模型。
- 图文任务要求恰好两个 `.rknn`，另一个是视觉模型。
- 每个 `.rknn` 必须配套同名前缀的 `.weight`；任务 JSON 文件名可自定义。

## 输入输出示例

以下为上述命令的板端实际结果，输入内容为演示样例。

### 客服分流

任务固定判断请求类别和紧急程度，更换用户消息即可得到对应结果：

| 用户输入 | 分类结果 | 分类概率 | 紧急程度评分（0～2） |
|---|---|---|---|
| 我无法登录账号，请帮我恢复访问。 | account：账号问题 | 95.48% | 0.9078 |
| 我被重复扣款了，请帮我处理。 | payment：支付问题 | 94.63% | 1.2708 |

评分是各等级按概率加权的结果；终端同时显示问题、候选选项及概率最高的等级。

### 街景判断

输入图片：

![街景图片](model/bus.jpg)

终端输出：

```text
输入：
  请根据图片回答问题。
  图片: model/bus.jpg

问题 [vehicle]: 图片中主要的大型车辆是什么？
  bus: 公交车或客车
  train: 火车
  boat: 船
  other: 其他车辆或没有上述车辆
结果: bus（公交车或客车），概率 82.46%

问题 [people]: 图片中是否有人？
  no: 没有人
  yes: 有人
结果: yes（有人），概率 75.72%
肯定概率: 75.72%
```
