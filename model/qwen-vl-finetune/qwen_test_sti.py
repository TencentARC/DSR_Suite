#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-2.5-VL end-to-end pipeline
1. Load dataset from qa.parquet                                 (read stage)
2. Compute sample FPS for up to 30 frames & run inference       (inference stage)
3. Save results to JSON and generate accuracy visualisations    (analysis stage)
"""

# --------------------------------------------------------------------------- #
#                               Imports & Setup                               #
# --------------------------------------------------------------------------- #
import os, re, json, signal, warnings
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info   # comes from Qwen repo
from qwenvl.train.qwen_vl_spatial import (
    Qwen2_5_VLForConditionalGeneration_Spatial,
)
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")                      # nicer plots

# --------------------------------------------------------------------------- #
#                               Configuration                                 #
# --------------------------------------------------------------------------- #
MODEL_ID     = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH   = "PATH_TO_MODEL"
PARQUET_FILE = "PATH_TO_STI_BENCH_PARQUET"
VIDEO_DIR    = "PATH_TO_STI_BENCH_VIDEO"
RESULTS_JSON = "finetuned.json"
DEVICE       = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
MAX_FRAMES   = 32   # target frames for time alignment

# --------------------------------------------------------------------------- #
#                            Timeout (per sample)                             #
# --------------------------------------------------------------------------- #
class TimeoutException(Exception): ...

def _timeout_handler(signum, frame): raise TimeoutException()
signal.signal(signal.SIGALRM, _timeout_handler)

# --------------------------------------------------------------------------- #
#                   Compute sample_fps without re-encoding                    #
# --------------------------------------------------------------------------- #
def compute_sample_fps(video_path: str, max_frames: int = MAX_FRAMES) -> float:
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

# --------------------------------------------------------------------------- #
#                      Robust answer extraction utility                       #
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
#                           Scene categorisation                              #
# --------------------------------------------------------------------------- #
def scene_type(filename: str) -> str:
    if "camera" in filename:
        return "outdoor"
    if filename.startswith("scene"):
        return "indoor"
    if filename.split(".")[0].isdigit() and len(filename.split(".")[0]) == 6:
        return "desktop"
    return "unknown"

# --------------------------------------------------------------------------- #
#                        Load model & processor                               #
# --------------------------------------------------------------------------- #
print(f"[Init] Loading model: {MODEL_ID}")
model = Qwen2_5_VLForConditionalGeneration_Spatial.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map=DEVICE
)
processor = Qwen2_5_VLProcessor.from_pretrained(
    MODEL_PATH, local_files_only=True, trust_remote_code=True
)
print("[Init] Model & processor ready.\n")

# --------------------------------------------------------------------------- #
#                          Read dataset from parquet                          #
# --------------------------------------------------------------------------- #
print(f"[Data] Reading {PARQUET_FILE}")
df_parquet = pd.read_parquet(PARQUET_FILE)        # remove .head(10) for full run
items = df_parquet.to_dict(orient="records")

for it in items:
    it["file"] = it.get("Video", "unknown.mp4")
    if isinstance(it.get("Candidates"), str):
        try: it["Candidates"] = json.loads(it["Candidates"])
        except json.JSONDecodeError: it["Candidates"] = {}
print(f"[Data] {len(items)} records loaded.\n")

# --------------------------------------------------------------------------- #
#                    Resume previous results if they exist                    #
# --------------------------------------------------------------------------- #
if os.path.exists(RESULTS_JSON):
    try:
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[Resume] Loaded {len(results)} previous results.\n")
    except Exception:
        results = []
else: results = []

# --------------------------------------------------------------------------- #
#                               Inference loop                                #
# --------------------------------------------------------------------------- #
def add_after_consecutive(tensor, target, count=32):
    i = 0
    tensor = tensor[0]
    while i < len(tensor) - 1:
        if tensor[i] == target and tensor[i + 1] != target:
            tensor = torch.cat([tensor[:i+1], torch.tensor([target] * count), tensor[i+1:]])
            i += count  # 跳过已经插入的部分
            break
        else:
            i += 1
    return tensor

print("[Run] Starting inference …")
for idx, entry in enumerate(items, 1):
    vid_name, sample_id = entry["file"], entry.get("ID")

    if any(r["id"] == vid_name and r["id_file"] == sample_id for r in results):
        print(f"[Skip] {idx}/{len(items)}  ({vid_name}, ID={sample_id})")
        continue

    print(f"[Run ] {idx}/{len(items)}  ({vid_name}, ID={sample_id})")
    video_path = os.path.join(VIDEO_DIR, vid_name)
    if not os.path.exists(video_path):
        print(f"       ! video not found → {video_path}")
        continue

    # prompt text
    cand_str = "\n".join([f"({k}) {v}" for k, v in entry["Candidates"].items()])
    ts, te = entry["time_start"], entry["time_end"]
    prompt_txt = (
        f"From {ts} s to {te} s. {entry['Question']}\n"
        f"{cand_str}\nPlease output only the option letter!"
    )

    # FPS for sampling
    try: sample_fps = compute_sample_fps(video_path, MAX_FRAMES)
    except Exception as e:
        print(f"       ! fps computation error: {e}")
        continue

    # multimodal message
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360*640,
                "fps": sample_fps
            },
            {"type": "text", "text": prompt_txt},
        ],
    }]

    signal.alarm(60)  # 60 s timeout
    try:
        template = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(
            text=[template], images=img_in, videos=vid_in,
            padding=True, return_tensors="pt"
        )
        inputs['input_ids'] = add_after_consecutive(inputs['input_ids'],151656).unsqueeze(0)
        inputs['attention_mask'] = torch.ones((1,inputs['attention_mask'].shape[1]+32)).to(inputs['attention_mask'].dtype)
        inputs = inputs.to(DEVICE)
        gen_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.6, do_sample=True, top_p=0.9, top_k=5)
        decoded = processor.batch_decode(
            [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        answer = extract_answer(decoded) or decoded.strip()
        print(f"       › raw: {decoded}")
        print(f"       › parsed answer: {answer}")

        results.append({
            "id": vid_name,
            "id_file": sample_id,
            "time_s": ts,
            "time_e": te,
            "scene": entry.get("scene") or entry.get("Scene"),  # 兼容大小写
            "model_out": answer,
            "ans": entry.get("Answer"),
            "Task": entry.get("Task"),
            "que": prompt_txt,
        })

    except TimeoutException:
        print("       ! timeout (>60 s)")
    except Exception as e:
        print(f"       ! error: {e}")
    finally:
        signal.alarm(0)
        torch.cuda.empty_cache()

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"[Save] {len(results)} results → {RESULTS_JSON}\n")

print(f"[Done] Inference completed — total saved: {len(results)}\n")

# --------------------------------------------------------------------------- #
#                                ANALYSIS                                     #
# --------------------------------------------------------------------------- #
print("[Ana ] Generating statistics & plots …")
out_dir = f"analysis_results_{MODEL_ID.replace('/','_')}"
os.makedirs(out_dir, exist_ok=True)

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    df = pd.DataFrame(json.load(f))

if "scene" not in df.columns:
    df["scene"] = df["id"].apply(scene_type)
else:
    df["scene"] = df["scene"].fillna(df["id"].apply(scene_type))

# --- correctness 列 -------------------------------------------------------- #
df["correct"] = df.apply(
    lambda r: (r["model_out"][-1] in "ABCDE") and (r["model_out"][-1] == r["ans"]),
    axis=1,
)

# -------------- radar charts (overall + per scene) ------------------------- #
tasks = sorted(df["Task"].unique())
task_acc_overall = df.groupby("Task")["correct"].mean().reindex(tasks, fill_value=0)
task_cnt_overall = df.groupby("Task")["correct"].count().reindex(tasks, fill_value=0)

def plot_radar(ax, labels, values, title, counts):
    filtered = [(l, v, c) for l, v, c in zip(labels, values, counts) if c > 0]
    if not filtered:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no samples)")
        return

    labels, values, _ = zip(*filtered)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    vals   = list(values) + [values[0]]
    angs   = list(angles) + [angles[0]]

    ax.plot(angs, vals, marker="o")
    ax.fill(angs, vals, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([.2,.4,.6,.8,1.0])
    ax.set_ylim(0, 1)
    ax.set_title(title, y=1.1)

scenes = ["indoor", "desktop", "outdoor"]
fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True), figsize=(14, 12))
plot_radar(axes[0, 0], tasks, task_acc_overall, "Overall", task_cnt_overall)
for ax, scn in zip(axes.flat[1:], scenes):
    sub = df[df["scene"] == scn]
    acc = sub.groupby("Task")["correct"].mean().reindex(tasks, fill_value=0)
    cnt = sub.groupby("Task")["correct"].count().reindex(tasks, fill_value=0)
    plot_radar(ax, tasks, acc, scn.capitalize(), cnt)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "radar_charts.png"), dpi=300)
plt.close()

# -------------- bar plot: correct vs total for each scene ------------------ #
correct_cnt = df.groupby("scene")["correct"].sum().reindex(scenes, fill_value=0)
total_cnt   = df.groupby("scene")["correct"].count().reindex(scenes, fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))
idx = np.arange(len(scenes))
bar_w = 0.35
ax.bar(idx, correct_cnt, bar_w, label="Correct")
ax.bar(idx + bar_w, total_cnt, bar_w, label="Total")

for i, (c, t) in enumerate(zip(correct_cnt, total_cnt)):
    acc = c / t if t else 0
    ax.text(i + bar_w/2, t + 1, f"{acc:.1%}", ha="center")

ax.set_xticks(idx + bar_w/2)
ax.set_xticklabels([s.capitalize() for s in scenes])
ax.set_title("Correct / Total by Scene")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "scene_counts.png"), dpi=300)
plt.close()

# -------------- textual report --------------------------------------------- #
overall_acc = df["correct"].mean()
task_acc  = df.groupby("Task")["correct"].mean().sort_values(ascending=False)
scene_acc = df.groupby("scene")["correct"].mean().sort_values(ascending=False)

report = [
    "=== Performance Report ===",
    f"Total samples: {len(df)}",
    f"Overall accuracy: {overall_acc:.2%}\n",
    "Accuracy by task:",
] + [f"  {t}: {a:.2%}" for t, a in task_acc.items()] + [
    "\nAccuracy by scene:",
] + [f"  {s}: {a:.2%}" for s, a in scene_acc.items()]

report_txt = "\n".join(report)
print(report_txt)

with open(os.path.join(out_dir, "analysis_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_txt)

print(f"[Ana ] Artifacts saved under: {out_dir}")
print("[Done] All stages complete.")
