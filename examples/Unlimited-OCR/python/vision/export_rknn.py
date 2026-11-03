import json
import os

import numpy as np
from rknn.api import RKNN


VISION_ONNX_MODEL = "../../model/vision/Unlimited-OCR-vision.onnx"
VISION_RKNN_MODEL = "../../model/vision/Unlimited-OCR-vision.rknn"
SAM_ONNX_MODEL = "../../model/vision/Unlimited-OCR-sam.onnx"
SAM_RKNN_MODEL = "../../model/vision/Unlimited-OCR-sam.rknn"
VISION_CONFIG = "vision_config.json"
DATASET_PATH = None
SAM_DATASET_PATH = None


def load_config(config_path):
    if not os.path.exists(config_path):
        return {
            "mode": "base",
            "base_size": 1024,
            "image_size": 1024,
            "crop_mode": False,
            "crop_width_num": 1,
            "crop_height_num": 1,
            "patch_size": 16,
            "downsample_ratio": 4,
            "hidden_size": 1280,
            "sam_hidden_size": 1024,
        }

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("mode", "gundam" if config.get("crop_mode") else "base")
    config.setdefault("base_size", 1024)
    config.setdefault("image_size", 1024 if config["mode"] == "base" else 640)
    config.setdefault("crop_mode", config["mode"] == "gundam")
    config.setdefault("crop_width_num", 1)
    config.setdefault("crop_height_num", 1)
    config.setdefault("patch_size", 16)
    config.setdefault("downsample_ratio", 4)
    config.setdefault("hidden_size", 1280)
    config.setdefault("sam_hidden_size", 1024)
    return config


def apply_cli_overrides(config, args):
    if args.mode is not None:
        config["mode"] = args.mode
        config["crop_mode"] = args.mode == "gundam"
    if args.base_size is not None:
        config["base_size"] = args.base_size
    if args.image_size is not None:
        config["image_size"] = args.image_size
    if args.crop_width_num is not None:
        config["crop_width_num"] = args.crop_width_num
    if args.crop_height_num is not None:
        config["crop_height_num"] = args.crop_height_num

    if config["mode"] == "base":
        config["crop_mode"] = False
        config["crop_width_num"] = 1
        config["crop_height_num"] = 1
    else:
        config["crop_mode"] = True
        if config["crop_width_num"] < 1 or config["crop_height_num"] < 1:
            raise ValueError("gundam mode requires positive crop_width_num/crop_height_num")
    return config


def sam_grid(size, config):
    return int(size) // int(config["patch_size"]) // int(config["downsample_ratio"])


def sam_feature_shape(batch, size, config):
    grid = sam_grid(size, config)
    return [batch, int(config["sam_hidden_size"]), grid, grid]


def resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def infer_raw_dtype(path):
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".u8", ".uint8"):
        return np.uint8
    if suffix in (".i8", ".int8"):
        return np.int8
    if suffix in (".fp16", ".float16"):
        return np.float16
    if suffix in (".fp32", ".float32"):
        return np.float32
    raise ValueError(
        f"Cannot infer raw dtype from {path!r}. Use .npy or a suffix like .u8/.fp16/.fp32."
    )


def load_accuracy_tensor_from_shapes(path, candidate_shapes):
    path = resolve_path(path)
    if path is None:
        return None, None
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".npy"):
        tensor = np.load(path)
    else:
        dtype = infer_raw_dtype(path)
        tensor = np.fromfile(path, dtype=dtype)

    matches = []
    for shape in candidate_shapes:
        expected = int(np.prod(shape))
        if tensor.size == expected:
            matches.append(shape)
    if not matches:
        raise ValueError(
            f"Accuracy input {path} has {tensor.size} elems, expected one of "
            f"{[(shape, int(np.prod(shape))) for shape in candidate_shapes]}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Accuracy input {path} matches multiple shapes {matches}; use a less ambiguous input size."
        )
    shape = matches[0]
    tensor = np.ascontiguousarray(tensor.reshape(shape))
    print(f"Loaded accuracy input: {path}, shape={tensor.shape}, dtype={tensor.dtype}")
    return tensor, shape


def load_accuracy_tensor(path, shape):
    tensor, _ = load_accuracy_tensor_from_shapes(path, [shape])
    return tensor


def run_accuracy_analysis(rknn, args, name, inputs, input_size_list, shape_idx=None):
    if not args.accuracy:
        return
    if args.device_id is None:
        raise ValueError("--accuracy requires --device_id for board-side inference accuracy analysis")
    if len(inputs) != len(input_size_list):
        raise ValueError(f"{name} accuracy inputs mismatch: {len(inputs)} vs {len(input_size_list)}")

    accuracy_dir = resolve_path(args.accuracy_output_dir)
    output_dir = os.path.join(accuracy_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"--> Init runtime for {name} accuracy analysis")
    ret = rknn.init_runtime(
        target=args.target or args.platform,
        device_id=args.device_id,
        core_mask=args.core_mask,
    )
    if ret != 0:
        print(f"Init runtime for {name} accuracy analysis failed!")
        exit(ret)
    print("done")

    print(f"--> Inference accuracy analysis {name}")
    outputs = rknn.inference(
        inputs=inputs,
        data_format=["nchw"] * len(inputs),
        output_dir=output_dir,
        accuracy_analysis=True,
        analysis_core_mask=args.core_mask,
        shape_idx=shape_idx,
    )
    if outputs is not None:
        try:
            print(f"Accuracy analysis {name} outputs: {[x.shape for x in outputs]}")
        except TypeError:
            print(f"Accuracy analysis {name} outputs: {outputs}")
    print(f"done {output_dir}")


def build_vision_input_spec(config):
    mode = config["mode"]
    base_size = int(config["base_size"])
    image_size = int(config["image_size"])
    crop_width_num = int(config["crop_width_num"])
    crop_height_num = int(config["crop_height_num"])

    if mode == "base":
        inputs = ["image_ori_sam_features"]
        input_size_list = [
            sam_feature_shape(1, image_size, config),
        ]
    elif mode == "gundam":
        crop_num = crop_width_num * crop_height_num
        inputs = [
            "patches_sam_features",
            "image_ori_sam_features",
        ]
        input_size_list = [
            sam_feature_shape(crop_num, image_size, config),
            sam_feature_shape(1, base_size, config),
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return inputs, input_size_list


def build_sam_input_spec(config):
    image_size = int(config["image_size"])
    base_size = int(config["base_size"])
    crop_num = int(config["crop_width_num"]) * int(config["crop_height_num"])

    if config["mode"] == "base":
        return ["pixel_values"], [[1, 3, image_size, image_size]], None

    shapes = [[crop_num, 3, image_size, image_size], [1, 3, base_size, base_size]]
    dynamic_input = []
    for shape in shapes:
        if shape not in (item[0] for item in dynamic_input):
            dynamic_input.append([shape])
    return None, None, dynamic_input


def expected_output_shape(config):
    base_grid = sam_grid(config["base_size"], config)
    hidden_size = int(config["hidden_size"])

    if config["mode"] == "base":
        token_num = base_grid * (base_grid + 1) + 1
    else:
        local_grid = sam_grid(config["image_size"], config)
        crop_width_num = int(config["crop_width_num"])
        crop_height_num = int(config["crop_height_num"])
        local_tokens = (crop_height_num * local_grid) * (crop_width_num * local_grid + 1)
        global_tokens = base_grid * (base_grid + 1)
        token_num = local_tokens + global_tokens + 1

    return [token_num, hidden_size]


def print_model_spec(config, vision_inputs, vision_input_size_list, sam_inputs, sam_input_size_list, sam_dynamic_input):
    print("Using Unlimited-OCR vision config:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("Vision RKNN inputs:")
    for name, shape in zip(vision_inputs, vision_input_size_list):
        print(f"  {name}: {shape}")
    print(f"Expected vision output image_embeds: {expected_output_shape(config)}")
    if sam_dynamic_input is None:
        print("SAM RKNN fixed inputs:")
        for name, shape in zip(sam_inputs, sam_input_size_list):
            print(f"  {name}: {shape}")
    else:
        print("SAM RKNN dynamic inputs:")
        for shape_group in sam_dynamic_input:
            print(f"  pixel_values: {shape_group[0]}")


def make_rknn(
    args,
    dynamic_input=None,
    input_attrs=None,
    mean_values=None,
    std_values=None,
    quantized_dtype=None,
    quantized_algorithm=None,
    quantized_method=None,
    do_quantization=None,
):
    if do_quantization is None:
        do_quantization = not args.no_quantization
    quantized_dtype = quantized_dtype or args.quantized_dtype
    quantized_algorithm = quantized_algorithm or args.quantized_algorithm
    quantized_method = quantized_method or args.quantized_method

    rknn = RKNN(verbose=True)
    print("--> config model")
    if do_quantization:
        print(
            "quantization: dtype={}, algorithm={}, method={}".format(
                quantized_dtype, quantized_algorithm, quantized_method
            )
        )
    else:
        print("quantization: disabled")
    config_kwargs = {
        "target_platform": args.platform,
        "core_num": args.core_num,
        "dynamic_input": dynamic_input,
        "profile_mode": True,
    }
    if do_quantization:
        config_kwargs.update(
            {
                "quantized_dtype": quantized_dtype,
                "quantized_algorithm": quantized_algorithm,
                "quantized_method": quantized_method,
            }
        )
    if input_attrs is not None:
        config_kwargs["input_attrs"] = input_attrs
    if mean_values is not None:
        config_kwargs["mean_values"] = mean_values
    if std_values is not None:
        config_kwargs["std_values"] = std_values
    rknn.config(**config_kwargs)
    print("done")
    return rknn

def sam_preprocess_config(args):
    if args.sam_input_dtype == "uint8":
        return {
            "input_attrs": {"pixel_values": {"dtype": "uint8", "layout": "NHWC"}},
            "mean_values": [[127.5, 127.5, 127.5]],
            "std_values": [[127.5, 127.5, 127.5]],
        }
    return {
        "input_attrs": {"pixel_values": {"dtype": args.sam_input_dtype, "layout": "NCHW"}},
        "mean_values": None,
        "std_values": None,
    }


def export_one_model(
    args,
    name,
    onnx_path,
    rknn_path,
    dataset_path,
    inputs=None,
    input_size_list=None,
    dynamic_input=None,
    input_attrs=None,
    mean_values=None,
    std_values=None,
    accuracy_inputs=None,
    accuracy_shape_idx=None,
    quantized_dtype=None,
    quantized_algorithm=None,
    quantized_method=None,
    do_quantization=None,
):
    print(f"==> Exporting {name}")
    if do_quantization is None:
        do_quantization = not args.no_quantization
    rknn = make_rknn(
        args,
        dynamic_input=dynamic_input,
        input_attrs=input_attrs,
        mean_values=mean_values,
        std_values=std_values,
        quantized_dtype=quantized_dtype,
        quantized_algorithm=quantized_algorithm,
        quantized_method=quantized_method,
        do_quantization=do_quantization,
    )

    if args.rebuild:
        print("--> Rebuilding model")
        ret = rknn.rebuild(args.rebuild_dir)
        if ret != 0:
            print(f"Rebuild {name} rknn model failed!")
            exit(ret)
        print("done")
    else:
        print("--> Loading model")
        if dynamic_input is None:
            ret = rknn.load_onnx(
                model=onnx_path,
                inputs=inputs,
                input_size_list=input_size_list,
                input_initial_val=[None] * len(inputs),
            )
        else:
            ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            print(f"Load {name} model failed!")
            exit(ret)
        print("done")

        print("--> Building model")
        ret = rknn.build(
            do_quantization=do_quantization,
            dataset=dataset_path if do_quantization else None,
        )
        if ret != 0:
            print(f"Build {name} model failed!")
            exit(ret)
        print("done")

    print("--> Export rknn model")
    export_rknn_dirname = os.path.dirname(rknn_path)
    if export_rknn_dirname and not os.path.exists(export_rknn_dirname):
        os.makedirs(export_rknn_dirname, exist_ok=True)

    ret = rknn.export_rknn(rknn_path, save_ctx=False)
    if ret != 0:
        print(f"Export {name} rknn model failed! {rknn_path}")
        exit(ret)
    print(f"done {rknn_path}")

    if accuracy_inputs is not None:
        run_accuracy_analysis(
            rknn,
            args,
            name,
            accuracy_inputs,
            input_size_list,
            shape_idx=accuracy_shape_idx,
        )

    rknn.release()


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Unlimited-OCR split vision convert rknn")
    parser.add_argument("--part", choices=["all", "vision", "sam"], default="all")
    parser.add_argument("--onnx_path", type=str, default=VISION_ONNX_MODEL)
    parser.add_argument("--rknn_path", type=str, default=VISION_RKNN_MODEL)
    parser.add_argument("--sam_onnx_path", type=str, default=SAM_ONNX_MODEL)
    parser.add_argument("--sam_rknn_path", type=str, default=SAM_RKNN_MODEL)
    parser.add_argument("--config_path", type=str, default=VISION_CONFIG)
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--sam_dataset_path", type=str, default=SAM_DATASET_PATH)
    parser.add_argument(
        "--sam_input_dtype",
        choices=["uint8", "float16", "float32"],
        default="uint8",
        help="SAM RKNN public input dtype. Default uint8 follows Qwen3_VL-style RKNN input preprocessing.",
    )
    parser.add_argument("--platform", type=str, default="rk1820")
    parser.add_argument("--core_num", type=int, default=8)
    parser.add_argument("--quantized_dtype", type=str, default="w4a16", help="Vision/CLIP quantized dtype.")
    parser.add_argument("--quantized_algorithm", type=str, default="normal", help="Vision/CLIP RKNN quantized algorithm. The ONNX can already be externally GRQ-quantized.")
    parser.add_argument("--quantized_method", type=str, default="group32", help="Vision/CLIP quantized method.")
    parser.add_argument("--sam_quantized_dtype", type=str, default="w16a16", help="Deprecated: SAM RKNN build is not quantized.")
    parser.add_argument("--sam_quantized_algorithm", type=str, default="normal", help="Deprecated: SAM RKNN build is not quantized.")
    parser.add_argument("--sam_quantized_method", type=str, default="group32", help="Deprecated: SAM RKNN build is not quantized.")
    parser.add_argument("--mode", choices=["base", "gundam"], default=None)
    parser.add_argument("--base_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--crop_width_num", type=int, default=None)
    parser.add_argument("--crop_height_num", type=int, default=None)
    parser.add_argument("--no_quantization", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--rebuild_dir", type=str, default="./tmp")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--device_id", type=str, default=None, help="RKNN board/device id for inference accuracy analysis.")
    parser.add_argument("--target", type=str, default=None, help="inference accuracy runtime target, defaults to --platform.")
    parser.add_argument("--core_mask", type=lambda x: int(x, 0), default=0xff, help="NPU core mask for inference accuracy analysis.")
    parser.add_argument("--accuracy_output_dir", type=str, default="./accuracy_snapshot")
    parser.add_argument("--sam_accuracy_input", type=str, default=None, help="SAM accuracy input, .npy or raw .u8/.fp16/.fp32.")
    parser.add_argument(
        "--vision_accuracy_input",
        action="append",
        default=None,
        help="Vision accuracy input for each vision ONNX input, in order. Repeat for multi-input gundam.",
    )
    args = parser.parse_args()

    config = apply_cli_overrides(load_config(args.config_path), args)
    vision_inputs, vision_input_size_list = build_vision_input_spec(config)
    sam_inputs, sam_input_size_list, sam_dynamic_input = build_sam_input_spec(config)
    print_model_spec(
        config,
        vision_inputs,
        vision_input_size_list,
        sam_inputs,
        sam_input_size_list,
        sam_dynamic_input,
    )

    sam_accuracy_inputs = None
    sam_accuracy_shape_idx = None
    if args.accuracy and args.part in ("all", "sam"):
        if args.sam_accuracy_input is None:
            raise ValueError("--accuracy for SAM requires --sam_accuracy_input")
        sam_candidate_shapes = sam_input_size_list or [shape_group[0] for shape_group in sam_dynamic_input]
        sam_accuracy_tensor, sam_accuracy_shape = load_accuracy_tensor_from_shapes(
            args.sam_accuracy_input, sam_candidate_shapes
        )
        sam_accuracy_inputs = [sam_accuracy_tensor]
        if sam_input_size_list is None:
            for idx, shape_group in enumerate(sam_dynamic_input):
                if shape_group[0] == sam_accuracy_shape:
                    sam_accuracy_shape_idx = idx
                    break
            sam_input_size_list = [sam_accuracy_shape]

    vision_accuracy_inputs = None
    if args.accuracy and args.part in ("all", "vision"):
        if not args.vision_accuracy_input:
            raise ValueError("--accuracy for vision requires --vision_accuracy_input for each vision input")
        if len(args.vision_accuracy_input) != len(vision_input_size_list):
            raise ValueError(
                f"Need {len(vision_input_size_list)} --vision_accuracy_input values, got {len(args.vision_accuracy_input)}"
            )
        vision_accuracy_inputs = [
            load_accuracy_tensor(path, shape)
            for path, shape in zip(args.vision_accuracy_input, vision_input_size_list)
        ]

    if args.part in ("all", "sam"):
        sam_cfg = sam_preprocess_config(args)
        export_one_model(
            args,
            name="sam",
            onnx_path=args.sam_onnx_path,
            rknn_path=args.sam_rknn_path,
            dataset_path=args.sam_dataset_path,
            inputs=sam_inputs,
            input_size_list=sam_input_size_list,
            dynamic_input=sam_dynamic_input,
            accuracy_inputs=sam_accuracy_inputs,
            accuracy_shape_idx=sam_accuracy_shape_idx,
            do_quantization=False,
            **sam_cfg,
        )

    if args.part in ("all", "vision"):
        export_one_model(
            args,
            name="vision",
            onnx_path=args.onnx_path,
            rknn_path=args.rknn_path,
            dataset_path=args.dataset_path,
            inputs=vision_inputs,
            input_size_list=vision_input_size_list,
            accuracy_inputs=vision_accuracy_inputs,
        )


if __name__ == "__main__":
    main()
