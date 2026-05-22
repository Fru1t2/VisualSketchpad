import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

base_dir = "/mnt/ssd2/cvlab_intern/VisualSketchpad/outputs"
experiments = [
    "eval_results_LoRA",
    "eval_results_LoRA_reply1",
    "eval_results_sketchpad",
    "eval_results_sketchpad_reply1"
]

tasks = ["blink_depth", "blink_spatial", "blink_jigsaw"]

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

# ==========================================
# 2. 3.0 Sigma 아웃라이어 제거 및 Clean Latency 추출
# ==========================================
avg_latencies = {task: {} for task in tasks}
SIGMA_THRESHOLD = 3.0

print(f"🔍 {SIGMA_THRESHOLD} Sigma 기준으로 아웃라이어 필터링 적용 중...\n")

for task in tasks:
    print(f"[{task}] 데이터 처리 중...")
    for exp in experiments:
        search_pattern = os.path.join(base_dir, exp, task, "*", "usage_summary.json")
        usage_files = glob.glob(search_pattern)
        
        if not usage_files:
            print(f"  경고: {exp}/{task} 경로에 파일이 없습니다.")
            continue
            
        raw_latencies = []
        
        # 모든 샘플의 Latency(초)를 리스트에 수집
        for file_path in usage_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    latency_sec = data.get("total", {}).get("latency", 0) / 1000.0
                    raw_latencies.append(latency_sec)
            except Exception as e:
                pass 
                
        if len(raw_latencies) > 0:
            arr = np.array(raw_latencies)
            
            # 통계 기반 상한선(Upper Bound) 계산
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            upper_bound = mean_val + (SIGMA_THRESHOLD * std_val)
            
            # 아웃라이어 제거 (상한선 이하인 값들만 남김)
            clean_arr = arr[arr <= upper_bound]
            removed_count = len(arr) - len(clean_arr)
            
            if len(clean_arr) > 0:
                clean_avg = np.mean(clean_arr)
                avg_latencies[task][exp] = clean_avg
                print(f"  - {exp:<30}: Clean 평균 {clean_avg:.2f}초 (총 {len(arr)}개 중 {removed_count}개 아웃라이어 제거)")

# ==========================================
# 3. 파레토 프론트(Pareto Front) 그리기
# ==========================================
labels_map = {
    "eval_results_LoRA": "LoRA (Base)",
    "eval_results_LoRA_reply1": "LoRA (1-iter)",
    "eval_results_sketchpad": "Sketchpad (Base)",
    "eval_results_sketchpad_reply1": "Sketchpad (1-iter)"
}
colors = ['blue', 'cyan', 'red', 'orange']

for task in tasks:
    x_latency = []
    y_accuracy = []
    labels = []
    
    for exp in experiments:
        if exp in avg_latencies[task] and exp in accuracies[task]:
            x_latency.append(avg_latencies[task][exp])
            y_accuracy.append(accuracies[task][exp])
            labels.append(labels_map[exp])
            
    if not x_latency:
        print(f"⚠️ {task}의 그래프를 그릴 데이터가 부족합니다.")
        continue

    plt.figure(figsize=(9, 6))

    for i in range(len(labels)):
        color_idx = list(labels_map.values()).index(labels[i]) 
        plt.scatter(x_latency[i], y_accuracy[i], color=colors[color_idx], s=150, zorder=5, label=labels[i])
        plt.annotate(labels[i], (x_latency[i], y_accuracy[i]), textcoords="offset points", xytext=(10,-5), fontsize=11)

    sorted_points = sorted(zip(x_latency, y_accuracy))
    pareto_x = []
    pareto_y = []
    max_acc = -1

    for lat, acc in sorted_points:
        if acc > max_acc:
            pareto_x.append(lat)
            pareto_y.append(acc)
            max_acc = acc

    plt.plot(pareto_x, pareto_y, '--', color='gray', linewidth=2, zorder=1, label='Pareto Front')

    # 제목과 X축 라벨에 Clean Latency임을 명시
    plt.title(f'Clean Latency vs Performance Trade-off ({task})', fontsize=15, pad=15)
    plt.xlabel(f'Clean Average Latency (Seconds) - 3.0 Sigma Filtered', fontsize=12)
    plt.ylabel('Accuracy (%) - Higher is Better', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.8)
    plt.legend(loc='lower right')

    plt.tight_layout()

    # 파일명 변경 (clean_pareto_front_...)
    save_path = f"clean_3.0_pareto_front_{task}.png"
    plt.savefig(save_path, dpi=300)
    plt.close() 
    print(f"📊 [{task}] 아웃라이어가 제거된 파레토 그래프가 '{save_path}'로 저장되었습니다!")