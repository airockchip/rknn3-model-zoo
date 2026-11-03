#!/usr/bin/env python3
"""Generate GRQ calibration data for Mage-VL Qwen3-4B LLM."""

import json
import os
import pickle
import sys

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_ZOO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(MODEL_ZOO)

from py_utils.cali_data import capture_module_input


MODEL_PATH = "microsoft/Mage-VL"
DATA_PATH = "../../../../datasets/MMBench/llm/dataset.json"
EXPORT_DATA_PATH = os.path.join(os.path.dirname(__file__), "quant_data/model_inputs.json")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate Mage-VL Qwen3-4B GRQ calibration data")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Mage-VL model path")
    parser.add_argument("--datapath", type=str, default=DATA_PATH, help="calibration source dataset path")
    parser.add_argument("--export_datapath", type=str, default=EXPORT_DATA_PATH, help="output GRQ dataset index path")
    parser.add_argument("--limit", type=int, default=256, help="maximum number of calibration samples")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    with open(args.datapath, "r", encoding="utf-8") as file:
        if args.datapath.lower().endswith(".jsonl"):
            datasets = [json.loads(line) for line in file if line.strip()]
        else:
            datasets = json.load(file)
    if args.limit > 0:
        datasets = datasets[:args.limit]

    info = []
    dataset_dir = os.path.dirname(os.path.abspath(args.datapath))
    export_dir = os.path.dirname(os.path.abspath(args.export_datapath))
    sample_dir = os.path.join(export_dir, "model_inputs")
    os.makedirs(sample_dir, exist_ok=True)

    for idx, data in enumerate(tqdm(datasets)):
        image_path = os.path.join(dataset_dir, data.get("image_path", ""), data["image"])
        prompt = data.get("input", data.get("prompt"))
        if not prompt:
            raise KeyError("Calibration sample {} has no input or prompt field".format(idx))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[Image.open(image_path).convert("RGB")], return_tensors="pt")
        inputs = {name: value.to(model.device) if hasattr(value, "to") else value for name, value in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        with torch.inference_mode():
            sample = capture_module_input(
                model.model.language_model,
                lambda: model.generate(**inputs, max_new_tokens=128),
            )
        if sample is None:
            raise RuntimeError("Failed to capture language model input for sample {}".format(idx))

        sample_name = "sample_{}".format(idx)
        with open(os.path.join(sample_dir, sample_name), "wb") as file:
            pickle.dump(sample, file)
        info.append({"sample": "model_inputs/{}".format(sample_name)})

    with open(args.export_datapath, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2, ensure_ascii=False)
    print("Generated {} LLM calibration samples to {}".format(len(info), args.export_datapath))
