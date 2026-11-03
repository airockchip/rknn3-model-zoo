import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input
import torch
from qwen_asr import Qwen3ASRModel
import pickle
import shutil

if __name__ == '__main__':
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        dtype=torch.bfloat16,
        device_map="cpu",
        max_inference_batch_size=1, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=512, # Maximum number of tokens to generate. Set a larger value for long audio input.
    )

    export_datapath = "./quant_data/model_input.json"
    case_index = 0
    info = []
    fp = open(f"../../../../datasets/ASR/audio_data.txt", "r")
    data = fp.readlines()
    fp.close()

    for d in data:
        with torch.inference_mode():
            temp = capture_module_input(
                model.model.thinker.model,
                lambda: model.transcribe(
                    audio=[os.path.join("../../../../datasets/ASR", d.strip())],
                    language=[None], # can also be set to None for automatic language detection
                    return_time_stamps=False,
                )
            )
        sample_name = "sample_{}".format(case_index)
        path_ = "/".join(export_datapath.split("/")[:-1])
        path_dir = os.path.join(path_, "model_inputs")
        os.makedirs(path_dir, exist_ok=True)
        pickle_path = os.path.join(path_dir, sample_name)
        info.append({"sample":"model_inputs/{}".format(sample_name)})
        with open(pickle_path, 'wb') as f:
            pickle.dump(temp, f)
        case_index += 1

    import random
    random.shuffle(info)

    with open(export_datapath, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
