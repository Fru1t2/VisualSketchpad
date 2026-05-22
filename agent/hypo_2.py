import os
import json
import glob
import numpy as np
import pandas as pd

base_dir = "/mnt/ssd2/cvlab_intern/VisualSketchpad/outputs"
experiments = [
    "eval_results_sketchpad",
    "eval_results_sketchpad_reply1",
    "eval_results_LoRA",
    "eval_results_LoRA_reply1"
]
tasks = ["blink_depth", "blink_spatial", "blink_jigsaw"]

# 🎯 시그마 임계치 설정 (추천: 2.0)
SIGMA_THRESHOLD = 3.0

results_list = []

print(f"🔍 {SIGMA_THRESHOLD}-Sigma 기준으로 아웃라이어 제거 중...\n")

for task in tasks:
    for exp in experiments:
        search_pattern = os.path.join(base_dir, exp, task, "*", "usage_summary.json")
        usage_files = glob.glob(search_pattern)
        
        if not usage_files:
            continue
            
        latencies = []
        
        # 1. 모든 Latency 데이터 수집
        for file_path in usage_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    latency_sec = data.get("total", {}).get("latency", 0) / 1000.0
                    latencies.append(latency_sec)
            except Exception:
                pass 
                
        if not latencies:
            continue
            
        # 2. 통계치 계산 (Numpy 활용)
        latencies_arr = np.array(latencies)
        mean_lat = np.mean(latencies_arr)
        std_lat = np.std(latencies_arr)
        
        # 3. Upper Bound (상한선) 계산: 평균 + (N * 표준편차)
        upper_bound = mean_lat + (SIGMA_THRESHOLD * std_lat)
        
        # 4. 아웃라이어 필터링 (상한선 이하인 정상 데이터만 추출)
        clean_latencies = latencies_arr[latencies_arr <= upper_bound]
        
        removed_count = len(latencies_arr) - len(clean_latencies)
        
        # 5. Clean 통계 재계산
        if len(clean_latencies) > 0:
            clean_avg = np.mean(clean_latencies)
            clean_max = np.max(clean_latencies)
        else:
            clean_avg, clean_max = 0, 0
            
        model_name = exp.replace("eval_results_", "")
        
        results_list.append({
            "Task": task,
            "Model": model_name,
            "Raw Avg (s)": round(mean_lat, 2),
            "Upper Bound": round(upper_bound, 2), # 이 시간 넘어가면 잘림
            "Removed": removed_count,             # 잘려나간 불량 샘플 수
            "Clean Avg (s)": round(clean_avg, 2), # 🌟 최종 사용할 깔끔한 평균
            "Clean Max (s)": round(clean_max, 2)
        })

# 데이터프레임 변환 및 출력
df = pd.DataFrame(results_list)
df = df.sort_values(by=["Task", "Model"])

print(f"📊 [{SIGMA_THRESHOLD} Sigma 아웃라이어 제거 후 Clean Latency 분석]")
print("-" * 90)
print(df.to_string(index=False))
print("-" * 90)