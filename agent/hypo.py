import os
import json
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

base_dir = "/mnt/ssd2/cvlab_intern/VisualSketchpad/outputs/eval_results_LoRA_reply1"

# 경로 형태: base_dir / task_name / sample_name / output.json
search_pattern = os.path.join(base_dir, "*", "*", "output.json")
output_files = glob.glob(search_pattern)

total_iterations = []
task_iterations = {} 

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

for file_path in output_files:
    parts = file_path.split(os.sep)
    task_name = parts[-3] 
    
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)

            assistant_texts = extract_assistant_text(data)
            combined_text = " ".join(assistant_texts)
            
            matches = re.findall(r"THOUGHT\s+(\d+):", combined_text)
            
            if matches:
                max_iter = int(max(matches, key=int))
            else:
                max_iter = 0
                
            total_iterations.append(max_iter)
            
            if task_name not in task_iterations:
                task_iterations[task_name] = []
            task_iterations[task_name].append(max_iter)
            
        except Exception as e:
            print(f" Error reading {file_path}: {e}")

# ==========================================
# 4. 분석 결과 DataFrame 변환 및 표 출력
# ==========================================
# 4-1. 전체 분포
counter_total = Counter(total_iterations)
df_total = pd.DataFrame(list(counter_total.items()), columns=['Iteration (반복)', 'Count (문제 수)'])
df_total = df_total.sort_values(by='Iteration (반복)').reset_index(drop=True)
df_total['Percentage (%)'] = (df_total['Count (문제 수)'] / len(total_iterations) * 100).round(2)

print("📊 [전체 데이터셋 반복 횟수 분포]")
print("-" * 40)
print(df_total.to_string(index=False))
print("-" * 40 + "\n")

print("📊 [태스크(Task)별 평균 반복 횟수]")
for task, iters in task_iterations.items():
    avg_iter = sum(iters) / len(iters)
    print(f" - {task:<15}: 평균 {avg_iter:.2f}회 (총 {len(iters)}문제)")

plt.figure(figsize=(10, 6))
bars = plt.bar(df_total['Iteration (반복)'].astype(str), df_total['Percentage (%)'], color='coral', edgecolor='black')

# 막대 위에 % 수치 표시
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontweight='bold')

plt.title('Distribution of Reasoning Iterations (All Tasks)', fontsize=14, pad=15)
plt.xlabel('Number of Iterations (Max THOUGHT step)', fontsize=12)
plt.ylabel('Percentage of Tasks (%)', fontsize=12)
plt.ylim(0, max(df_total['Percentage (%)']) + 15) # Y축 공간 넉넉히
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
save_path = "iteration_distribution_plot_LoRA_reply1.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight') 