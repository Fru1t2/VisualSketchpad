import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. 경로 및 설정
base_dir = "/mnt/ssd2/cvlab_intern/VisualSketchpad/outputs"
experiments = [
    "eval_results_LoRA",
    "eval_results_LoRA_reply1",
    "eval_results_sketchpad",
    "eval_results_sketchpad_reply1"
]
tasks = ["blink_depth", "blink_spatial", "blink_jigsaw"]

# 정확도 데이터 (기존과 동일)
accuracies = {
    "blink_depth": {
        "eval_results_LoRA": 86.29,             
        "eval_results_LoRA_reply1": 80.65,      
        "eval_results_sketchpad": 79.84,        
        "eval_results_sketchpad_reply1": 80.65  
    },
    "blink_spatial": {
        "eval_results_LoRA": 86.01,             
        "eval_results_LoRA_reply1": 86.01,      
        "eval_results_sketchpad": 79.02,        
        "eval_results_sketchpad_reply1": 79.72 
    },
    "blink_jigsaw": {
        "eval_results_LoRA": 62.67,             
        "eval_results_LoRA_reply1": 58.00,      
        "eval_results_sketchpad": 68.67,        
        "eval_results_sketchpad_reply1": 65.33
    }
}

# 2. 1.5 Sigma 아웃라이어 제거 및 Clean Latency 추출
avg_latencies = {task: {} for task in tasks}
SIGMA_THRESHOLD = 3.0


for task in tasks:
    for exp in experiments:
        search_pattern = os.path.join(base_dir, exp, task, "*", "usage_summary.json")
        usage_files = glob.glob(search_pattern)
        
        raw_latencies = []
        for file_path in usage_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    latency_sec = json.load(f).get("total", {}).get("latency", 0) / 1000.0
                    raw_latencies.append(latency_sec)
            except Exception:
                pass 
                
        if len(raw_latencies) > 0:
            arr = np.array(raw_latencies)
            mean_val, std_val = np.mean(arr), np.std(arr)
            upper_bound = mean_val + (SIGMA_THRESHOLD * std_val)
            clean_arr = arr[arr <= upper_bound]
            
            if len(clean_arr) > 0:
                avg_latencies[task][exp] = np.mean(clean_arr)

# 3. 통합 그래프(Combined Plot) 시각화 세팅
# 마커(모양)는 태스크를 구분
markers = {
    "blink_depth": "o",   # 원형
    "blink_spatial": "s", # 사각형
    "blink_jigsaw": "^"   # 삼각형
}
# 색상은 모델을 구분
colors = {
    "eval_results_LoRA": "blue",
    "eval_results_LoRA_reply1": "cyan",
    "eval_results_sketchpad": "red",
    "eval_results_sketchpad_reply1": "orange"
}
labels_map = {
    "eval_results_LoRA": "LoRA (Base)",
    "eval_results_LoRA_reply1": "LoRA (1-iter)",
    "eval_results_sketchpad": "Sketchpad (Base)",
    "eval_results_sketchpad_reply1": "Sketchpad (1-iter)"
}
task_names = {"blink_depth": "Depth", "blink_spatial": "Spatial", "blink_jigsaw": "Jigsaw"}

plt.figure(figsize=(12, 8)) # 여러 개가 들어가니 캔버스를 조금 더 크게!

# 4. 데이터 플로팅 및 태스크별 파레토 선 긋기
for task in tasks:
    x_lat, y_acc, model_labels = [], [], []
    
    for exp in experiments:
        if exp in avg_latencies[task] and exp in accuracies[task]:
            x = avg_latencies[task][exp]
            y = accuracies[task][exp]
            
            # 산점도(Scatter) 그리기
            plt.scatter(x, y, color=colors[exp], marker=markers[task], s=200, edgecolor='black', zorder=5)
            
            x_lat.append(x)
            y_acc.append(y)
            model_labels.append(labels_map[exp])
            
    # 해당 태스크의 파레토 프론트 계산 및 그리기
    sorted_points = sorted(zip(x_lat, y_acc))
    pareto_x, pareto_y = [], []
    max_acc = -1

    for lat, acc in sorted_points:
        if acc > max_acc:
            pareto_x.append(lat)
            pareto_y.append(acc)
            max_acc = acc

    # 태스크별로 파레토 선 그리기 (알아보기 쉽게 투명도 조절)
    plt.plot(pareto_x, pareto_y, '--', color='gray', linewidth=2, alpha=0.6, zorder=1)
    
    # 파레토 선 끝에 어떤 태스크의 선인지 작게 텍스트 달아주기
    if pareto_x:
        plt.text(pareto_x[-1] + 0.3, pareto_y[-1], f"{task_names[task]} Front", color='gray', fontweight='bold')

# 5. 커스텀 범례(Legend) 만들기
legend_elements = [
    # 모양(태스크) 범례
    Line2D([0], [0], marker='none', color='w', label='[Task Type]'),
    Line2D([0], [0], marker='o', color='w', label='Depth', markerfacecolor='gray', markersize=12),
    Line2D([0], [0], marker='s', color='w', label='Spatial', markerfacecolor='gray', markersize=12),
    Line2D([0], [0], marker='^', color='w', label='Jigsaw', markerfacecolor='gray', markersize=12),
    # 색상(모델) 범례
    Line2D([0], [0], marker='none', color='w', label='\n[Model Type]'),
    Line2D([0], [0], marker='o', color='w', label='LoRA (Base)', markerfacecolor='blue', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='LoRA (1-iter)', markerfacecolor='cyan', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='Sketchpad (Base)', markerfacecolor='red', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='Sketchpad (1-iter)', markerfacecolor='orange', markersize=12),
]

plt.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.25, 0.5), fontsize=11)

# 6. 스타일링 및 마무리
plt.title('Combined Clean Latency vs Performance Trade-off', fontsize=16, pad=15)
plt.xlabel('Clean Average Latency (Seconds) - 3.0 Sigma Filtered', fontsize=13)
plt.ylabel('Accuracy (%) - Higher is Better', fontsize=13)
plt.grid(True, linestyle=':', alpha=0.8)

# 범례가 잘리지 않도록 여백 조정
plt.tight_layout(rect=[0, 0, 0.85, 1]) 

save_path = "clean_pareto_front_combined.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"📊 [통합] 3개 태스크가 합쳐진 파레토 그래프가 '{save_path}'로 저장되었습니다!")