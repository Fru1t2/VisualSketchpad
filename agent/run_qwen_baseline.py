from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from PIL import Image
import torch
import json
import re
import os
import time

GREEDY = os.environ.get('greedy', 'false').lower() == 'true'
TOP_P = float(os.environ.get('top_p', 0.8))
TOP_K = int(os.environ.get('top_k', 20))
TEMPERATURE = float(os.environ.get('temperature', 0.7))
REPETITION_PENALTY = float(os.environ.get('repetition_panalty', 1.0))
PRESENCE_PENALTY = float(os.environ.get('presence_penalty', 1.5))
OUT_SEQ_LENGTH = int(os.environ.get('out_seq_length', 16384))


def parse_choice(text):
    matches = re.findall(r"\(([A-E])\)", text)
    if matches:
        return matches[-1]
    
    matches = re.findall(r"\b([A-E])\b", text)
    if matches:
        return matches[-1]
        
    return None

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", dtype=torch.float16, device_map="cuda:0"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

tasks = ["blink_depth", "blink_spatial", "blink_jigsaw"]
base_tasks_dir = "/mnt/ssd2/cvlab_intern/VisualSketchpad/tasks"
final_report = {}

for task_name in tasks:
    print(f"\n Starting Task : {task_name}")
    processed_dir = os.path.join(base_tasks_dir, task_name, "processed")

    if not os.path.exists(processed_dir):
        print(f"Directory not found : {processed_dir}")
        continue
    
    results = []
    correct_count = 0
    sample_dirs = sorted([d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))])

    for sample_name in tqdm(sample_dirs, desc=f"Processing {task_name}"):
        sample_path = os.path.join(processed_dir, sample_name)
        req_file = os.path.join(sample_path, "request.json")

        if not os.path.exists(req_file): continue

        with open(req_file, "r") as f:
            data = json.load(f)
        img_filename = os.path.basename(data["images"][0])
        img_abs_path = os.path.join(sample_path, img_filename)
        image_obj = Image.open(img_abs_path).convert("RGB")
        
        query = re.sub(r"<imag[^>]*>", "", data["query"]).strip()
        # query = query + "\nAnswer with the option letter form the given choices directly."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_obj},
                    {"type": "text", "text": query},
                ],
            }
        ]

        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=OUT_SEQ_LENGTH,
            do_sample=not GREEDY,         
            top_p=TOP_P,
            top_k=TOP_K,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000

        input_length = inputs.input_ids.shape[1]
        output_ids = generated_ids[0, input_length:]
        token_count = len(output_ids)

        output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        prediction = parse_choice(output_text)
        gt_answer = parse_choice(data["answer"])
        is_correct = (prediction == gt_answer)
        if is_correct: correct_count += 1

        results.append({
            "id": sample_name,
            "prediction_raw": output_text,
            "prediction_parsed": prediction,
            "ground_truth": gt_answer,
            "is_correct": is_correct,
            "latency": latency,
            "token_count": token_count
        })

    accuracy = (correct_count / len(results)) * 100 if results else 0
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0
    avg_tokens = sum(r["token_count"] for r in results) / len(results) if results else 0
    
    final_report[task_name] = {
        "accuracy": f"{accuracy:.2f}%", 
        "total": len(results), 
        "correct": correct_count,
        "avg_latency": avg_latency,
        "avg_tokens": avg_tokens
    }

    with open(f"eval_{task_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

print("\n" + "="*80)
print(f"{'Task Name':<15} | {'Accuracy':<10} | {'Score':<10} | {'Avg Latency (s)':<15} | {'Avg Tokens'}")
print("-" * 80)
for task, score in final_report.items():
    print(f"{task:<15} | {score['accuracy']:<10} | {score['correct']}/{score['total']:<10} | {score['avg_latency']:<15.2f} | {score['avg_tokens']:.1f}")
print("="*80)

# with open(os.path.join(task_dir, "request.json"), "r") as f:
#     data = json.load(f)

# query_pretained = data["query"]
# query = re.sub(r"<imag[^>]*>", "", query_pretained).strip()

# image = data["images"][0]
# answer = data["answer"]

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": image,
#             },
#             {"type": "text", "text": query},
#         ],
#     }
# ]

# # Preparation for inference
# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt"
# )
# inputs = inputs.to(model.device)

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)