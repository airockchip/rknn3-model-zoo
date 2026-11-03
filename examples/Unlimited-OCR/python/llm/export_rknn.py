import copy
import json
import os
import pickle
from argparse import ArgumentParser

from rknn.api import DEFAULT_RKNN_LLM_CONFIG, RKNN


ONNX_MODEL = "../../model/llm/Unlimited-OCR-llm.onnx"
LLM_CONFIG = "../../model/llm/Unlimited-OCR-llm.config.pkl"
RKNN_MODEL = "../../model/llm/Unlimited-OCR-llm.rknn"
DATASET_PATH = None


def build_dynamic_input(prefill_seq_len):
    return [
        [[1, 1], [1, 1], [1, 1], [1]],
        [[1, prefill_seq_len], [1, prefill_seq_len], [1, prefill_seq_len], [1]],
    ]


def build_llm_config(max_ctx_len, kvcache_buffer_len, attention, sliding_window):
    llm_config = copy.deepcopy(DEFAULT_RKNN_LLM_CONFIG)
    attention_config = llm_config["attention_config"][0]
    attention_config.update(
        {
            "kvcache_buffer_len": kvcache_buffer_len,
            "max_position_embeddings": max_ctx_len,
        }
    )
    if attention == "sliding":
        attention_config.update(
            {
                "attention_type": "SlidingAttention",
                "sliding_window_size": sliding_window,
            }
        )
    else:
        attention_config["attention_type"] = "FullAttention"
        attention_config.pop("sliding_window_size", None)
    llm_config["llm_head"][0].update(
        {"quantized_dtype": "w6a16", "quantized_method": "group32"}
    )
    return llm_config


def prepare_llm_config(config_path, rknn_path, attention, sliding_window):
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    if not isinstance(config, dict):
        raise TypeError(f"Unsupported LLM config type: {type(config)!r}")

    config = copy.deepcopy(config)
    hf_config = json.loads(config.get("hf_config_json", "{}"))
    num_hidden_layers = int(hf_config["num_hidden_layers"])
    layer_type = "sliding_attention" if attention == "sliding" else "full_attention"
    hf_config["layer_types"] = [layer_type] * num_hidden_layers
    if attention == "sliding":
        hf_config["sliding_window"] = sliding_window
        hf_config["sliding_window_size"] = sliding_window
    else:
        hf_config.pop("sliding_window", None)
        hf_config.pop("sliding_window_size", None)
    config["hf_config_json"] = json.dumps(hf_config, ensure_ascii=False)

    out_dir = os.path.dirname(os.path.abspath(rknn_path))
    os.makedirs(out_dir, exist_ok=True)
    rknn_stem = os.path.splitext(os.path.basename(rknn_path))[0]
    load_config = os.path.join(out_dir, f"{rknn_stem}.{attention}.config.pkl")
    with open(load_config, "wb") as f:
        pickle.dump(config, f)
    return load_config


def main():
    parser = ArgumentParser(description="Export Unlimited-OCR LLM ONNX to RKNN model.")
    parser.add_argument("--onnx_path", type=str, default=ONNX_MODEL, help="Input LLM ONNX model path.")
    parser.add_argument("--config", type=str, default=LLM_CONFIG, help="LLM config pkl path.")
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL, help="Output RKNN model path.")
    parser.add_argument("--platform", type=str, default="rk1820", help="Target platform, e.g. rk1820.")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="Quantization dataset path.")
    parser.add_argument("--core_num", type=int, default=8, help="NPU core number, range depends on target platform.")
    parser.add_argument(
        "--max_ctx_len",
        type=int,
        default=8192,
        help="Maximum position/context length. Unlimited-OCR config uses 32768.",
    )
    parser.add_argument(
        "--kvcache_buffer_len",
        type=int,
        default=8192,
        help="KV cache buffer length used by RKNN LLM runtime.",
    )
    parser.add_argument(
        "--attention",
        choices=("sliding", "full"),
        default="sliding",
        help="Attention mode used by RKNN LLM conversion.",
    )
    parser.add_argument(
        "--sliding_window",
        type=int,
        default=128,
        help="SlidingAttention window size. Unlimited-OCR uses 128. Ignored when --attention full.",
    )
    parser.add_argument(
        "--prefill_seq_len",
        type=int,
        default=128,
        help="Static prefill length used in RKNN dynamic_input. Increase if build needs a larger prefill profile.",
    )
    parser.add_argument("--no_quant", dest="quant", action="store_false", help="Disable RKNN build quantization.")
    parser.set_defaults(quant=True)
    args = parser.parse_args()

    if args.attention == "sliding" and args.sliding_window <= 0:
        raise ValueError("--sliding_window must be positive when --attention sliding")

    os.makedirs(os.path.dirname(os.path.abspath(args.rknn_path)), exist_ok=True)

    dynamic_input = build_dynamic_input(args.prefill_seq_len)
    llm_config = build_llm_config(
        args.max_ctx_len,
        args.kvcache_buffer_len,
        args.attention,
        args.sliding_window,
    )
    load_config = prepare_llm_config(
        args.config,
        args.rknn_path,
        args.attention,
        args.sliding_window,
    )

    rknn = RKNN(verbose=True)

    print("--> Config model")
    rknn.config(
        target_platform=args.platform,
        core_num=args.core_num,
        rknn_build_parallelism=64,
        profile_mode=False,
        dynamic_input=dynamic_input,
        llm_config=llm_config,
        quantized_dtype="w4a16",
        quantized_algorithm="normal",
        quantized_method="group32",
    )
    print("attention:", args.attention)
    print("dynamic_input:", dynamic_input)
    print("seq_lens:", [1, args.prefill_seq_len])
    print("load_config:", load_config)
    print("done")

    print("--> Loading model")
    ret = rknn.load_llm(model=args.onnx_path, config=load_config, seq_lens=[1, args.prefill_seq_len])
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    print("--> Building model")
    ret = rknn.build(do_quantization=args.quant, dataset=args.dataset_path)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    print("--> Export RKNN model")
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print("Export RKNN model failed!")
        exit(ret)
    print("done")

    rknn.release()


if __name__ == "__main__":
    main()
