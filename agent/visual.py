import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "/mnt/ssd2/cvlab_intern/VisualSketchpad/outputs"
experiment = "eval_results_LoRA" 
tasks = ["blink_depth", "blink_spatial", "blink_jigsaw"]

tool_keywords = {
    "Depth": ["depth("],
    "Sliding_window": ["sliding_window_detection("],
    "Detection": ["detection("],
    "Segmentation": ["segment_and_mark"],
    "Zoom": ["zoom_in_image_by_bbox("]
}

task_tool_counts = {task: {tool: 0 for tool in tool_keywords} for task in tasks}
task_totals = {task: 0 for task in tasks} 

def extract_assistant_text(data):
    texts = []
    if isinstance(data, list):
        for item in data:
            texts.extend(extract_assistant_text(item))
    elif isinstance(data, dict):
        if data.get("role") == "assistant":
            content = data.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        texts.append(c["text"])
                    elif isinstance(c, str):
                        texts.append(c)
            elif isinstance(content, str):
                texts.append(content)
        else:
            for key, value in data.items():
                texts.extend(extract_assistant_text(value))
    return texts

for task in tasks:
    search_pattern = os.path.join(base_dir, experiment, task, "*", "output.json")
    output_files = glob.glob(search_pattern)
    
    if not output_files:
        print(f" {task} 경로에 파일이 없습니다.")
        continue
        
    for file_path in output_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            assistant_texts = extract_assistant_text(data)
            combined_text = " ".join(assistant_texts)
            
            task_totals[task] += 1
            
            for tool_name, keywords in tool_keywords.items():
                is_used = False
                for kw in keywords:
                    if kw in combined_text:
                        is_used = True
                        break 
                        
                if is_used:
                    task_tool_counts[task][tool_name] += 1
                    
        except Exception as e:
            pass

df_data = []
for task in tasks:
    row = {"Task": task, "Total Samples": task_totals[task]}
    row.update(task_tool_counts[task])
    df_data.append(row)

df = pd.DataFrame(df_data)
print("Tool use Frequency")
print("-" * 80)
print(df.to_string(index=False))
print("-" * 80 + "\n")

plt.figure(figsize=(10, 6))

x = np.arange(len(tasks))
tools = list(tool_keywords.keys())

bottoms = np.zeros(len(tasks))

colors = ['#5b9bd5', '#ed7d31', '#70ad47', '#ffc000', '#9e480e']

for i, tool in enumerate(tools):
    counts = [task_tool_counts[task][tool] for task in tasks]
    
    plt.bar(x, counts, bottom=bottoms, label=tool, color=colors[i % len(colors)], edgecolor='white', width=0.6)
    
    bottoms += np.array(counts)

plt.title(f'Visual Sketchpad LoRA Tool-Use Frequency', fontsize=16, pad=15)
plt.xticks(x, tasks, fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)

plt.legend(title='Vision Experts', fontsize=11, title_fontsize=12, loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()

save_path = "tool_use_frequency_stacked_LoRA.png"
plt.savefig(save_path, dpi=300)