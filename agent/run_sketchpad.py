import json
import os
import re
import sys
import contextlib
import io
import time
import torch
from tqdm import tqdm
from main import run_agent

# target_samples = [
#     "val_Spatial_Relation_110", "val_Spatial_Relation_111", "val_Spatial_Relation_115",
#     "val_Spatial_Relation_120", "val_Spatial_Relation_121", "val_Spatial_Relation_125",
#     "val_Spatial_Relation_13", "val_Spatial_Relation_133", "val_Spatial_Relation_137",
#     "val_Spatial_Relation_16", "val_Spatial_Relation_19", "val_Spatial_Relation_29",
#     "val_Spatial_Relation_41", "val_Spatial_Relation_45", "val_Spatial_Relation_63",
#     "val_Spatial_Relation_67", "val_Spatial_Relation_9"
# ]

# def parse_choice(text):
#     if not text: return None

#     clean_text = text.strip().lower()
#     if clean_text == "yes": return "A"
#     if clean_text == "no": return "B"

#     match = re.search(r"\(([A-Ea-e])\)", text)
#     if match: return match.group(1).upper()
#     match = re.search(r"(?:answer is|ANSWER:)\s*([A-Ea-e])", text, re.IGNORECASE)
#     if match: return match.group(1).upper()
#     match = re.search(r"\b([A-Ea-e])\b", text)
#     if match: return match.group(1).upper()
#     return None

def parse_choice(text):
    if text is None: return None
    
    # [보완 1] 리스트나 숫자 타입이 들어와도 처리할 수 있게 강제 문자열 변환
    if isinstance(text, list):
        text = " ".join([str(i) for i in text])
    else:
        text = str(text)

    clean_text = text.lower().strip()

    match = re.search(r"answer:\s*(?:\()?([a-e]|yes|no)(?:\))?", clean_text)
    if match:
        ans = match.group(1)
        if ans == "yes": return "A"
        if ans == "no": return "B"
        return ans.upper()

    if clean_text == "yes": return "A"
    if clean_text == "no": return "B"

    match = re.search(r"[\(\[]([A-Ea-e])[\)\]]", text)
    if match: return match.group(1).upper()
    
    match = re.search(r"(?:answer is|answer:)\s*([A-Ea-e])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    match = re.search(r"\b([A-Ea-e])\b", text)
    if match: return match.group(1).upper()

    if "yes" in clean_text: return "A"
    if "no" in clean_text: return "B"
    
    return None

def extract_agent_answer(output_data):
    for msg in reversed(output_data):
        if isinstance(msg, dict) and 'content' in msg:
            content = msg['content']
            text_to_parse = ""
            if isinstance(content, str):
                text_to_parse = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_to_parse += item.get('text', "")
            parsed = parse_choice(text_to_parse)
            if parsed: return parsed, text_to_parse 
    return None, "No valid answer found"

tasks = ["blink_depth", "blink_spatial", "blink_jigsaw"]
base_tasks_dir = "/mnt/ssd2/cvlab_intern/VisualSketchpad/tasks"
# base_output_dir = "../outputs/eval_results_lora"
base_output_dir = "../outputs/eval_results_LoRA_reply1"
final_report = {}

for task_name in tasks:
    print(f"\n [Task Start] : {task_name}")
    processed_dir = os.path.join(base_tasks_dir, task_name, "processed")
    task_output_root = os.path.join(base_output_dir, task_name)
    if not os.path.exists(processed_dir): continue

    all_dirs = sorted([d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))])
    sample_dirs = all_dirs

    if not sample_dirs:
        print(f"Skipping {task_name}: No target samples found.")
        continue

    print(f"Found {len(sample_dirs)} samples in {task_name}. Starting inference...")
    # sample_dirs = sorted([d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))])
    correct_count = 0
    total_count = 0

    for sample_name in sample_dirs:
        sample_path = os.path.join(processed_dir, sample_name)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with contextlib.redirect_stdout(io.StringIO()):
            try:
                run_agent(sample_path, task_output_root, task_type="vision")
            except: pass

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # ms 단위로

        usage_path = os.path.join(task_output_root, sample_name, "usage_summary.json")

        if os.path.exists(usage_path):
            with open(usage_path, "r") as f:
                usage_data = json.load(f)
            
            if "total" in usage_data:
                usage_data["total"]["latency"] = latency
            else:
                usage_data["latency"] = latency

            with open(usage_path, "w") as f:
                json.dump(usage_data, f, indent=4)

        output_file = os.path.join(task_output_root, sample_name, "output.json")
        req_file = os.path.join(sample_path, "request.json")

        if os.path.exists(output_file) and os.path.exists(req_file):
            with open(output_file, "r") as f: agent_log = json.load(f)
            with open(req_file, "r") as f: gt_data = json.load(f)
            pred, raw_content = extract_agent_answer(agent_log)
            gt = parse_choice(gt_data["answer"])
            if pred == gt:
                correct_count += 1

            total_count += 1
            acc = (correct_count / total_count) * 100
            display_content = raw_content.replace('\n', ' ').strip()[-60:]
            print(f"[{sample_name}] 파싱: {pred} | 정답: {gt} | 정확도: {acc:.2f}% | 원본: ...{display_content}")

    accuracy_final = (correct_count / total_count) * 100 if total_count > 0 else 0
    final_report[task_name] = {"accuracy": f"{accuracy_final:.2f}%", "score": f"{correct_count}/{total_count}"}

print("\n" + "="*50)
print(f"{'Task Name':<20} | {'Accuracy':<12} | {'Score'}")
print("-" * 50)
for task, data in final_report.items():
    print(f"{task:<20} | {data['accuracy']:<12} | {data['score']}")
print("="*50)
