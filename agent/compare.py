import json
import os
import re
import sys
import contextlib
import io
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from main import run_agent

GREEDY = os.environ.get('greedy', 'false').lower() == 'true'
TOP_P = float(os.environ.get('top_p', 0.8))
TOP_K = int(os.environ.get('top_k', 20))
TEMPERATURE = float(os.environ.get('temperature', 0.7))
REPETITION_PENALTY = float(os.environ.get('repetition_panalty', 1.0))
PRESENCE_PENALTY = float(os.environ.get('presence_penalty', 1.5))
OUT_SEQ_LENGTH = int(os.environ.get('out_seq_length', 16384))

# 1. 설정 및 타겟 샘플
target_samples = [
    "val_Spatial_Relation_110", "val_Spatial_Relation_111", "val_Spatial_Relation_115",
    "val_Spatial_Relation_120", "val_Spatial_Relation_121", "val_Spatial_Relation_125",
    "val_Spatial_Relation_13", "val_Spatial_Relation_133", "val_Spatial_Relation_137",
    "val_Spatial_Relation_16", "val_Spatial_Relation_19", "val_Spatial_Relation_29",
    "val_Spatial_Relation_41", "val_Spatial_Relation_45", "val_Spatial_Relation_63",
    "val_Spatial_Relation_67", "val_Spatial_Relation_9"
]

tasks = ["blink_spatial"] 
base_tasks_dir = "/mnt/ssd2/cvlab_intern/VisualSketchpad/tasks"
base_output_dir_agent = "../outputs/eval_with_sketchpad"

print("Loading Baseline Model (Qwen3-VL)...")
model_id = "Qwen/Qwen3-VL-4B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# def parse_choice(text):
#     if not text: return None
#     match = re.search(r"\(([A-Ea-e])\)", text)
#     if match: return match.group(1).upper()
#     match = re.search(r"(?:answer is|ANSWER:)\s*([A-Ea-e])", text, re.IGNORECASE)
#     if match: return match.group(1).upper()
#     match = re.search(r"\b([A-Ea-e])\b", text)
#     if match: return match.group(1).upper()
#     return None

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

import re

def parse_choice(text):
    if text is None: return None
    
    # [보완 1] 리스트나 숫자 타입이 들어와도 처리할 수 있게 강제 문자열 변환
    if isinstance(text, list):
        text = " ".join([str(i) for i in text])
    else:
        text = str(text)

    clean_text = text.lower().strip()

    # 1순위: "ANSWER: yes" 혹은 "ANSWER: (B)" 등 명시적 포맷
    # 유저님의 정규식을 조금 더 확장해서 괄호가 포함된 경우도 한 번에 잡습니다.
    match = re.search(r"answer:\s*(?:\()?([a-e]|yes|no)(?:\))?", clean_text)
    if match:
        ans = match.group(1)
        if ans == "yes": return "A"
        if ans == "no": return "B"
        return ans.upper()

    # 2순위: 단답형 (yes / no)
    if clean_text == "yes": return "A"
    if clean_text == "no": return "B"

    # 3순위: (B) 혹은 [B] 형태 (괄호 포함)
    match = re.search(r"[\(\[]([A-Ea-e])[\)\]]", text)
    if match: return match.group(1).upper()
    
    # 4순위: "Answer is B" 형태
    match = re.search(r"(?:answer is|answer:)\s*([A-Ea-e])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # 5순위: 단독으로 쓰인 알파벳 (문장 끝이나 단어 경계)
    match = re.search(r"\b([A-Ea-e])\b", text)
    if match: return match.group(1).upper()

    # 6순위: 최후의 수단 (문장 안에 yes/no 단어가 있는지 확인)
    if "yes" in clean_text: return "A"
    if "no" in clean_text: return "B"
    
    return None


def extract_agent_answer(output_data):
    for msg in reversed(output_data):
        if isinstance(msg, dict) and 'content' in msg:
            content = msg['content']
            text_to_parse = ""
            if isinstance(content, str): text_to_parse = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_to_parse += item.get('text', "")
            parsed = parse_choice(text_to_parse)
            if parsed: return parsed
    return None

# 4. 메인 비교 루프
final_report = {}

for task_name in tasks:
    print(f"\n{'='*20} [Task Start] : {task_name} {'='*20}")
    processed_dir = os.path.join(base_tasks_dir, task_name, "processed")
    if not os.path.exists(processed_dir): continue

    all_dirs = os.listdir(processed_dir)
    # Target에 있고 실제 폴더가 존재하는 것만 필터링
    sample_dirs = sorted([d for d in all_dirs if d in target_samples and os.path.isdir(os.path.join(processed_dir, d))])
    
    stats = {"no_sketch": 0, "with_sketch": 0, "total": 0}
    
    for sample_name in tqdm(sample_dirs, desc=f"Comparing {task_name}"):
        sample_path = os.path.join(processed_dir, sample_name)
        req_file = os.path.join(sample_path, "request.json")
        if not os.path.exists(req_file): continue

        with open(req_file, "r") as f:
            data = json.load(f)
        gt = parse_choice(data["answer"])

        # --- A. No Sketchpad (Pure Model Inference) ---
        img_filename = os.path.basename(data["images"][0])
        image_obj = Image.open(os.path.join(sample_path, img_filename)).convert("RGB")
        query = re.sub(r"<imag[^>]*>", "", data["query"]).strip()

        messages = [{"role": "user", "content": [{"type": "image", "image": image_obj}, {"type": "text", "text": query}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=16384)

        output_text = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        pred_no = parse_choice(output_text)

        # baseline 결과 저장
        baseline_output_root = "../outputs/eval_no_sketchpad"
        baseline_sample_dir = os.path.join(baseline_output_root, task_name, sample_name)
        os.makedirs(baseline_sample_dir, exist_ok=True)

        with open(os.path.join(baseline_sample_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump({
                "sample_name": sample_name,
                "ground_truth": gt,
                "query": query,
                "raw_output": output_text,
                "parsed_choice": pred_no
            }, f, ensure_ascii=False, indent=2)

        # --- B. With Sketchpad (Agent Execution) ---
        output_root = os.path.join(base_output_dir_agent, task_name)
        with contextlib.redirect_stdout(io.StringIO()): 
            try:
                run_agent(sample_path, output_root, task_type="vision")
            except: pass
        
        output_file = os.path.join(output_root, sample_name, "output.json")
        pred_yes = None
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                agent_log = json.load(f)
            pred_yes = extract_agent_answer(agent_log)

        # 결과 카운트
        stats["total"] += 1
        if pred_no == gt: stats["no_sketch"] += 1
        if pred_yes == gt: stats["with_sketch"] += 1

        # 개별 결과 비교 출력
        print(f"\n[{sample_name}]")
        print(f"  └─ 정답: {gt} | 미적용(Baseline): {pred_no} | 적용(Sketchpad): {pred_yes}")

    final_report[task_name] = stats

# 5. 최종 리포트 출력
print("\n" + "="*75)
print(f"{'Task Name':<20} | {'No Sketchpad (Baseline)':<22} | {'With Sketchpad (Agent)':<22}")
print("-" * 75)
for task, s in final_report.items():
    if s["total"] == 0: continue
    acc_no = (s["no_sketch"] / s["total"]) * 100
    acc_yes = (s["with_sketch"] / s["total"]) * 100
    print(f"{task:<20} | {acc_no:>6.2f}% ({s['no_sketch']}/{s['total']}) | {acc_yes:>6.2f}% ({s['with_sketch']}/{s['total']})")
print("="*75)