import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from py_utils.export_vision_helper import export_qwen2_vl_vision, export_minicpmv4_7_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig
import torch
import torch.nn as nn



class Qwen3_5VisionExportWrapper(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        outputs = self.visual(
            pixel_values,
            grid_thw=grid_thw,
            return_dict=True,
        )
        return outputs.pooler_output


def _install_minicpmv4_7_compatibility():
    """Provide the activation alias used by MiniCPM's remote code.

    The cp310 Transformers installation has ``GELUTanh`` but not the older
    functional alias imported by the MiniCPM-V 4.7 modeling file.
    """
    import transformers.activations as activations

    if not hasattr(activations, "gelu_pytorch_tanh"):
        def gelu_pytorch_tanh(hidden_states):
            return torch.nn.functional.gelu(hidden_states, approximate="tanh")

        activations.gelu_pytorch_tanh = gelu_pytorch_tanh


def _load_minicpmv4_7_vision_only(model_path, config):
    """Instantiate and load only MiniCPM-V 4.7 visual weights.

    Loading the full 35B conditional-generation model just to export vision
    would unnecessarily consume tens of GB of host memory.  The safetensors
    index lets us copy only ``model.vision_tower.*`` and ``model.merger.*``.
    """
    from safetensors import safe_open
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"MiniCPM checkpoint index not found: {index_path}")

    vision_class = get_class_from_dynamic_module(
        "modeling_minicpmv4_7.MiniCPMV4_7VisionModel",
        model_path,
        local_files_only=True,
    )
    merger_class = get_class_from_dynamic_module(
        "modeling_minicpmv4_7.MiniCPMV4_7Merger",
        model_path,
        local_files_only=True,
    )

    # The vision encoder uses the eager attention path during ONNX tracing.
    if hasattr(config.vision_config, "_attn_implementation"):
        config.vision_config._attn_implementation = "eager"
    vision_model = vision_class._from_config(config.vision_config).float().eval()
    merger = merger_class(config).float().eval()

    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    def load_module_weights(module, checkpoint_prefix):
        state = module.state_dict()
        by_shard = {}
        missing = []
        for name in state:
            checkpoint_name = checkpoint_prefix + name
            shard_name = weight_map.get(checkpoint_name)
            if shard_name is None:
                missing.append(checkpoint_name)
            else:
                by_shard.setdefault(shard_name, []).append((checkpoint_name, name))

        if missing:
            raise RuntimeError(
                f"Missing {len(missing)} weights for {checkpoint_prefix}; "
                f"first missing key: {missing[0]}"
            )

        loaded = 0
        with torch.no_grad():
            for shard_name, names in by_shard.items():
                shard_path = os.path.join(model_path, shard_name)
                print(f"Loading {checkpoint_prefix} from {shard_name} ({len(names)} tensors)")
                with safe_open(shard_path, framework="pt", device="cpu") as shard:
                    for checkpoint_name, name in names:
                        state[name].copy_(shard.get_tensor(checkpoint_name))
                        loaded += 1
        print(f"Loaded {loaded} tensors for {checkpoint_prefix}")

    load_module_weights(vision_model, "model.vision_tower.")
    load_module_weights(merger, "model.merger.")
    return vision_model, merger

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export MiniCPM-V 4.7 vision model to ONNX for RKNN")
    parser.add_argument(
        "--model_path",
        type=str,
        help="MiniCPM-V 4.7 model path",
        required=False,
        default="/path/to/MiniCPM-V-4.7-35B-A3B",
    )
    parser.add_argument(
        "--export_vision_path",
        type=str,
        help="export vision ONNX model path",
        required=False,
        default="../../../model/vision/minicpm4_7_35b_a3b_vision/MiniCPM-V-4.7-35B-A3B_vision.onnx",
    )
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--img_h", type=int, help="Input image height; MiniCPM defaults to 448.", required=False, default=None)
    parser.add_argument("--img_w", type=int, help="Input image width; MiniCPM defaults to 448.", required=False, default=None)
    
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    if getattr(config, "model_type", None) == "minicpmv4_7":
        _install_minicpmv4_7_compatibility()
        args.img_h = args.img_h or int(getattr(config, "image_size", 448))
        args.img_w = args.img_w or int(getattr(config, "image_size", 448))
        print("Detected MiniCPM-V 4.7 checkpoint; loading vision_tower and merger only.")
        vision_model, merger = _load_minicpmv4_7_vision_only(args.model_path, config)

        export_vision_dirname = os.path.dirname(args.export_vision_path)
        if export_vision_dirname and not os.path.exists(export_vision_dirname):
            print(f"create export_vision_dirname: {export_vision_dirname}")
            os.makedirs(export_vision_dirname)

        export_minicpmv4_7_vision(vision_model, merger, args, config)
        raise SystemExit(0)

    # Qwen3.5 path kept unchanged for existing users of this script.
    from transformers import Qwen3_5ForConditionalGeneration
    args.img_h = args.img_h or 384
    args.img_w = args.img_w or 384
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float32,
        # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
        low_cpu_mem_usage=True, _attn_implementation="eager",
        trust_remote_code=True)
    
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.model.visual, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.model.visual = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)
        

        model = model.cpu()

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if export_vision_dirname and not os.path.exists(export_vision_dirname):
        print(f"create export_vision_dirname: {export_vision_dirname}")
        os.makedirs(export_vision_dirname)

    # patch_size 为 16；若 config 里有值则以 config 为准
    patch_size = 16
    if getattr(config, "vision_config", None) is not None and config.vision_config.patch_size is not None:
        patch_size = config.vision_config.patch_size
    else:
        print(f"vision_config.patch_size is None, use default value {patch_size}")

    vision_wrapper = Qwen3_5VisionExportWrapper(model.model.visual).eval()

    # export vision model
    export_qwen2_vl_vision(vision_wrapper, args, patch_size) # 添加grid_thw对输入图片的patch
