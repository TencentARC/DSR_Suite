from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd
from tqdm import tqdm
import json
import argparse
import torch.multiprocessing as mp
import torch
import cv2
import re
def compute_sample_fps(video_path: str, max_frames: int = 32) -> float:
    """Return FPS that yields ≤ max_frames evenly-spaced frames over the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / orig_fps if orig_fps else total / 1.0
    cap.release()

    if total <= max_frames or duration == 0:
        return orig_fps
    return max_frames / duration

ANSWER_PATTERNS = [
    r"\(([A-E])\)",                                # (A)
    r"Ans\s*=\s*['\"]?([A-E])['\"]?",              # Ans='C'
    r"Answer\s*[:=]\s*([A-E])",                    # Answer: B
    r"Option\s+([A-E])",                           # Option D
    r"\b([A-E])\s*(?:is|was)\s*correct",           # A is correct
    r"\b([A-E])[\.\)]\s*$",                        # C.  /  D)
]

def extract_answer(text: str) -> str | None:
    for pat in ANSWER_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m: return m.group(1).upper()
    return None
MODEL_PATH = "PATH_TO_QWEN_25"

def split_data(data, num_processes):
    avg_len = math.ceil(len(data) / num_processes)
    return [data[i:i + avg_len] for i in range(0, len(data), avg_len)]

def evaluate(physical_gpu_id, part_info, shared_dict):
    #device = torch.device(f"cuda:{physical_gpu_id}" if torch.cuda.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype="bfloat16", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    video_root = "PATH_TO_VLM4D_VIDEO"
    for qa in tqdm(part_info):
        filename = qa['video']
        sample_fps = compute_sample_fps(f"{video_root}/{filename}")
        question = qa['question']
        question += ". Choose the closest answer from the following 4 candidates and only output your choice as one of A,B,C,D: "
        question += "A: {}; B: {}; C: {}; D: {}".format(qa['choices']["A"],qa['choices']["B"],qa['choices']["C"],qa['choices']["D"])
        messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"{video_root}/{filename}",
                    "max_pixels": 360 * 640,
                    "fps":sample_fps,
                },
                {"type": "text", "text": question},
            ],
        }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        _, video_inputs = process_vision_info(messages)
        inputs = processor(
        text=[text],
        videos=video_inputs,
        num_frames=32,
        padding=True,
        return_tensors="pt",
        )
        def calculate_frame_interval(video_path, avg_frame_count):
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            frame_interval = video_duration / avg_frame_count
    
            return frame_interval
        inputs['second_per_grid_ts'] = [calculate_frame_interval(messages[0]['content'][0]['video'],inputs['video_grid_thw'][0,0]).item()]
        inputs = inputs.to("cuda")
        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=20, temperature=0.4, do_sample=True, top_p=0.9, top_k=5)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for i in ["A","B","C","D"]:
            if qa['answer']==qa['choices'][i]:
                break
        answer = extract_answer(output_text[0]) or output_text[0].strip()
        if answer==i:
            shared_dict['correct'] += 1
        shared_dict['count'] +=1

num_processes = 4
meta_info_real = json.load(open("PATH_TO_VLM4D_REAL_MC"))
for item in meta_info_real:
    video = item['video'].split("/")[-3:]
    video = "/".join(video)
    item['video'] = video
meta_info_sync = json.load(open("PATH_TO_VLM4D_SYNTHETIC_MC"))
for item in meta_info_sync:
    video = item['video'].split("/")[-2:]
    video = "/".join(video)
    item['video'] = video
meta_info_real.extend(meta_info_sync)
shared_dict = dict({
    "correct": 0,
    "count": 0
})
evaluate(0,meta_info_real,shared_dict)
accuracy = shared_dict['correct']/shared_dict['count']
with open(args.model.split("/")[-1]+".txt","w") as f:
    f.write(str(accuracy))
