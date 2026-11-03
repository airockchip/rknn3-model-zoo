import os
import sys
import json
import torch
import numpy as np
from transformers import AutoConfig

# 根据你的工程结构调整路径
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.tools import clear_llm_external_weight_in_dir

from transformers import AutoModelForCausalLM, AutoConfig




def _relpath_from_file(path, json_path):
    """把路径写成相对于 json 文件的路径，方便整套文件移动。"""
    return os.path.relpath(os.path.abspath(path), os.path.dirname(os.path.abspath(json_path)))


def _write_rknn_load_json(args, in_h, in_w, num_patches, fake_pos_ids):
    """保存 export_rknn.py 需要的 rknn.load_onnx 参数。"""
    export_vision_abspath = os.path.abspath(args.export_vision_path)
    export_dir = os.path.dirname(export_vision_abspath)

    json_path = args.rknn_load_json
    if json_path is None:
        json_path = os.path.join(export_dir, "gemma4_vision_rknn_load.json")
    json_path = os.path.abspath(json_path)
    json_dir = os.path.dirname(json_path)
    os.makedirs(json_dir, exist_ok=True)

    pos_ids_path = args.pixel_position_ids_path
    if pos_ids_path is None:
        pos_ids_path = os.path.join(export_dir, f"gemma4_pos_ids_{in_h}x{in_w}.npy")
    pos_ids_path = os.path.abspath(pos_ids_path)
    os.makedirs(os.path.dirname(pos_ids_path), exist_ok=True)

    pixel_position_ids = fake_pos_ids.cpu().numpy().astype(np.int64)
    np.save(pos_ids_path, pixel_position_ids)

    cfg = {
        "onnx_path": _relpath_from_file(export_vision_abspath, json_path),
        "pixel_position_ids_path": _relpath_from_file(pos_ids_path, json_path),
        "load_onnx": {
            "inputs": ["pixel", "pixel_position_ids"],
            "input_size_list": [[1, 3, in_h, in_w], [1, num_patches, 2]],
            # JSON 不能直接保存 ndarray，所以这里用 npy 文件描述第二个 initial value。
            "input_initial_val": [
                None,
                {"type": "npy", "path": _relpath_from_file(pos_ids_path, json_path)}
            ]
        }
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"Saved pixel_position_ids to {pos_ids_path}")
    print(f"Saved RKNN load json to {json_path}")


def export_gemma4_vision(model1, model2, args, patch_size=16):
    class Gemma4VisionWrapper(torch.nn.Module):
        def __init__(self, vision_tower, embed_vision, in_h, in_w, patch_size):
            super(Gemma4VisionWrapper, self).__init__()
            self.vision_tower = vision_tower
            self.embed_vision = embed_vision
            self.patch_size = patch_size
            
            # 根据要求，宽高必须是48的倍数对齐
            align_size = 48
            align_h = (in_h + align_size - 1) // align_size * align_size
            align_w = (in_w + align_size - 1) // align_size * align_size
            
            self.in_h = align_h
            self.in_w = align_w

        def forward(self, pixel, pixel_position_ids):
            pixel = pixel.float() 
            
            B, C, H, W = pixel.shape
            num_patches_h = H // self.patch_size
            num_patches_w = W // self.patch_size
            
            # 将 4D Tensor 转为 Patches
            # 1. Reshape -> [B, C, num_patches_h, patch_size, num_patches_w, patch_size]
            patched_image = pixel.reshape(B, C, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
            # 2. Permute -> [B, num_patches_h, num_patches_w, patch_size, patch_size, C]
            patched_image = patched_image.permute(0, 2, 4, 3, 5, 1).contiguous()
            # 3. Reshape 展平 -> [B, num_patches_h * num_patches_w, patch_size * patch_size * C]
            patched_image = patched_image.reshape(B, num_patches_h * num_patches_w, -1)

            last_hidden_state = self.vision_tower(patched_image, pixel_position_ids).last_hidden_state
            # 调用原始vision model，传入patch和位置ids
            return self.embed_vision(inputs_embeds=last_hidden_state)

    in_h = args.img_h
    in_w = args.img_w
    model = Gemma4VisionWrapper(model1, model2, in_h, in_w, patch_size)
    
    in_h = model.in_h
    in_w = model.in_w

    num_patches_h = in_h // patch_size
    num_patches_w = in_w // patch_size
    num_patches = num_patches_h * num_patches_w

    # 构建 Dummy Inputs: 图像和位置编码
    fake_input = torch.randint(0, 255, (1, 3, in_h, in_w), dtype=torch.float32)
    
    # 构造 pixel_position_ids: shape [1, max_patches, 2]
    y, x = torch.meshgrid(torch.arange(num_patches_h), torch.arange(num_patches_w), indexing='ij')
    fake_pos_ids = torch.stack([x, y], dim=-1).reshape(1, -1, 2).to(torch.int64)

    # 推理测试一次
    out = model(fake_input, fake_pos_ids)

    # 导出ONNX
    # 若存在check_torch_version()方法可类似原脚本加判断，这里提供通用的导出格式
    torch.onnx.export(
        model,
        (fake_input, fake_pos_ids),
        args.export_vision_path,
        input_names=['pixel', 'pixel_position_ids'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'pixel': {2: 'height', 3: 'width'},
            'pixel_position_ids': {1: 'max_patches'}
        },
        opset_version=18
    )
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")

    # 额外导出 export_rknn.py 需要读取的 npy + json
    _write_rknn_load_json(args, in_h, in_w, num_patches, fake_pos_ids)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Gemma4 vision configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="google/gemma-4-vision")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/Gemma4-vision.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--rknn_load_json", type=str, required=False, default="gemma4_vision_rknn_load.json",
                        help="Path to save RKNN load_onnx json. Default: same dir as gemma4_vision_rknn_load.json")
    parser.add_argument("--pixel_position_ids_path", type=str, required=False, default="gemma4_pos_ids.npy",
                        help="Path to save pixel_position_ids npy. Default: same dir as gemma4_pos_ids_HxW.npy")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--img_h", type=int, help="Input image height. Must be a multiple of 48.", required=False, default=384)
    parser.add_argument("--img_w", type=int, help="Input image width. Must be a multiple of 48.", required=False, default=384)
    
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    fakeq_state = None
    if args.quant and torch.cuda.is_available():
        
        kwargs = {
            'trust_remote_code': True,
        }
        config = AutoConfig.from_pretrained(args.model_path, **kwargs)
        kwargs['config'] = config
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
        
        from rknn.quantization.api import RKQuantizer

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.model.vision_tower, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.model.vision_tower = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)
        
        
        model = model.cpu()
        fakeq_state = model.state_dict()
        del model
        torch.cuda.empty_cache()
        
    # Get the custom module path
    custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gemma4'))
    print(f"Custom gemma4 module path: {custom_path}")

    # Load our custom modeling_gemma4 module using importlib
    import importlib.util
    modeling_spec = importlib.util.spec_from_file_location(
        'transformers.models.gemma4.modeling_gemma4',
        os.path.join(custom_path, 'modeling_gemma4.py')
    )
    modeling_module = importlib.util.module_from_spec(modeling_spec)
    sys.modules['transformers.models.gemma4.modeling_gemma4'] = modeling_module
    modeling_spec.loader.exec_module(modeling_module)
    print(f"Loaded custom modeling_gemma4 from: {modeling_module.__file__}")

    
    import transformers.models.gemma4 as _gemma4_pkg
    for _name in getattr(modeling_module, '__all__', []):
        if hasattr(modeling_module, _name):
            setattr(_gemma4_pkg, _name, getattr(modeling_module, _name))
    import transformers.models.auto as _auto_pkg
    for _attr in dir(_auto_pkg):
        _mapping = getattr(_auto_pkg, _attr, None)
        if hasattr(_mapping, '_modules') and isinstance(getattr(_mapping, '_modules', None), dict):
            _mapping._modules.pop('gemma4', None)
    
    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
        "torch_dtype": torch.float32
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    kwargs['config'] = config
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    if fakeq_state is not None:
        model.load_state_dict(
                fakeq_state,
                strict=False
            )
    
    export_vision_dirname = os.path.dirname(args.export_vision_path) or '.'
    if not os.path.exists(export_vision_dirname):
        os.makedirs(export_vision_dirname)

    # 导出 vision model (在Gemma4的架构中，vision module 为 model.model.vision_tower)
    # patch_size 指定为 16
    export_gemma4_vision(model.model.vision_tower, model.model.embed_vision, args, patch_size=16) 