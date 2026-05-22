import json
import os
import sys
import requests
import contextlib
import io
import threading  
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image 
from io import BytesIO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from main import run_agent
from datasets import load_from_disk

VSR_TRAIN_PATH = "./dataset/train"
TASKS_DIR = "./tasks/vsr_task/processed"
TASK_OUTPUT_ROOT = "./outputs/vsr_task"
LORA_OUTPUT_FILE = "./vsr_lora_train.jsonl"

file_lock = threading.Lock()

def get_processed_ids():
    processed_ids = set()
    if os.path.exists(LORA_OUTPUT_FILE):
        with open(LORA_OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["id"])
                except:
                    continue
    return processed_ids

def find_marked_image(sample_output_dir):
    if not os.path.exists(sample_output_dir):
        return None
    png_files = [os.path.join(sample_output_dir, f) for f in os.listdir(sample_output_dir) 
                 if f.endswith('.png') and f != "image.png"]
    if not png_files:
        return None
    return max(png_files, key=os.path.getmtime)

def process_sample(args):
    i, vsr_sample = args
    sample_name = f"vsr_train_{i:05d}"
    sample_path = os.path.join(TASKS_DIR, sample_name)
    output_dir = os.path.join(TASK_OUTPUT_ROOT, sample_name)
    
    try:
        os.makedirs(sample_path, exist_ok=True)

        img_filename = "image.jpg"
        img_full_path = os.path.join(sample_path, img_filename)

        if not os.path.exists(img_full_path):
            response = requests.get(vsr_sample['image_link'], timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(img_full_path)
        
        abs_img_path = os.path.abspath(img_full_path)

        request_data = {
            "query": vsr_sample['caption'],
            "images": [abs_img_path]
        }
        with open(os.path.join(sample_path, "request.json"), "w") as f:
            json.dump(request_data, f, indent=4)

        output_json_path = os.path.join(output_dir, "output.json")        
        marked_img_path = find_marked_image(output_dir)

        if not marked_img_path:
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    run_agent(sample_path, TASK_OUTPUT_ROOT, task_type="vision")
                except Exception as e:
                    return {"error": f"{sample_name} Agent Error: {str(e)}"}
            
            marked_img_path = find_marked_image(output_dir)
        if marked_img_path:
            return {
                "id": sample_name,
                "image": marked_img_path,
                "conversation": [
                    {
                        "from": "user",
                        "value": f"<image>\nBased on the visual aids, answer the question: {vsr_sample['caption']}? (A) Yes (B) No"
                    },
                    {
                        "from": "assistant",
                        "value": "Yes" if vsr_sample['label'] == 1 else "No"
                    }
                ]
            }
    except Exception as e:
        return {"error": f"{sample_name}: {str(e)}"}
    
    return None

def preprocess_vsr_parallel():
    dataset = load_from_disk(VSR_TRAIN_PATH)
    os.makedirs(os.path.dirname(LORA_OUTPUT_FILE), exist_ok=True)

    processed_ids = get_processed_ids()

    target_indices = [i for i in range(len(dataset)) if f"vsr_train_{i:05d}" not in processed_ids]

    MAX_WORKERS = 4 
    
    print(f"\n 병렬 처리 시작 (Workers: {MAX_WORKERS}, Total: {len(dataset)})")

    with open(LORA_OUTPUT_FILE, "a", encoding="utf-8", buffering=1) as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_sample, (idx, dataset[idx])) for idx in target_indices]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                
                if result and "id" in result:
                    with file_lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush() 
                elif result and "error" in result:
                    pass

if __name__ == "__main__":
    preprocess_vsr_parallel()