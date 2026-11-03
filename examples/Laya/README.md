# Laya RKNN3 部署指南

## 模型说明

Laya 根据输入状态回答选择、评分或真假判断问题。模型下载链接：[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)。

## 模型接口

选择预置任务，直接传入本次文本即可：

| 任务 | 判断内容 | 任务配置 |
| --- | --- | --- |
| `customer_service` | 账号／支付／其他类别、紧急程度 | [customer_service.json](model/customer_service.json) |
| `review_analysis` | 正面／中性／负面、满意程度 | [review_analysis.json](model/review_analysis.json) |

修改任务配置即可调整问题和候选项。

## Demo 流程

Laya 将输入文本与预设的问题、候选项一起编码，每个问题通过一次主图推理得到所有候选项的得分。Demo 逐个处理任务中的问题，并汇总答案：

```text
用户文本 + 任务配置（问题、类型、候选项）
                  │
                  ▼
CPU 组装当前问题的输入
分词 → 标记候选位置 → 自动选桶 → embedding 查表
                  │
                  ▼
NPU 执行 Laya 主图（每个问题一次）
ModernBERT → 决策 head → 候选项打分
                  │
                  ▼
输出 logits、p、k、h0
                  │
                  ▼
CPU 后处理
  ├─ logits → 温度校准 + Softmax → 选项概率 / 评分 / 真假判断
  └─ p、k、h0 → act head → act_probability
                  │
                  ▼
汇总各问题答案 → 终端展示 / 可选保存 JSON
```

`p` 为未经温度校准的候选概率，`k` 为有效选项数，`h0` 为决策 head 输出的首 token 特征。分类取概率最高的选项，评分按等级概率加权，真假判断取肯定选项的概率；`act_probability` 单独记录在结果 JSON 中。

## Python 环境

依赖安装和模型导出均在 `examples/Laya` 目录执行：

```bash
pip install -r requirements.txt
```

## 导出 ONNX

```bash
python python/export_onnx.py \
  --model_path convaiinnovations/laya \
  --output_path outputs/onnx
```

`--model_path` 接受本地模型目录或远端模型 ID；远端默认使用 Hugging Face，添加 `--modelscope` 可将所有远端来源切换为 ModelScope，本地目录不受影响。`--output_path` 为 ONNX 和配套资源的输出目录。

## 导出 RKNN

```bash
# --onnx_path 指定 ONNX 文件，配套资源需放在同一目录。
# --output_path 指定 RKNN 文件，自动生成同名 weight 并复制配套资源；输出目录需与 ONNX 目录不同。
python python/export_rknn.py \
  --onnx_path outputs/onnx/laya.onnx \
  --output_path model/laya.rknn
```

## 编译

在工程根目录执行以下命令，编译 Demo：

```bash
export GCC_COMPILER=/path/to/toolchain/bin/aarch64-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d Laya -b Release -r -j
```

## 板端运行

在工程根目录执行以下命令，将编译产物推送到板端：

```bash
adb push install/rk3588_linux_aarch64/rknn_Laya_demo /userdata/
```

在板端执行：

```bash
cd /userdata/rknn_Laya_demo
export LD_LIBRARY_PATH=./lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
./rknn_laya_demo --model_path model --task customer_service \
  --text "我无法登录账号，请帮我恢复访问。"

./rknn_laya_demo --model_path model --task review_analysis \
  --text "商品质量很好，物流也很快，非常满意，下次还会购买。"
```

`--model_path` 指向模型和任务配置目录，目录中需有唯一的 `.rknn` 文件及同名 `.weight` 文件。`--task` 选择任务，`--text` 指定输入文本。添加 `--output_path result.json` 可保存完整 JSON。

## 结果输出

终端显示输入、问题、候选项、判断结果及概率，评分为各等级的概率加权值。上述命令的输出示例：

| 任务 | 分类结果 | 评分 |
| --- | --- | --- |
| `customer_service` | account：账号登录或访问问题 | 紧急程度 1.7162（0～2） |
| `review_analysis` | positive：正面评价 | 满意程度 3.0660（0～4） |
