import os
import json
import random

parent_dir = "/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new/dpo_data/female/short/train-1200"
cer_threshold = 0.1
sim_threshold = 0.75
max_neg_per_pos = 1

all_pairs = []

for subdir in sorted(os.listdir(parent_dir)):
    subdir_path = os.path.join(parent_dir, subdir)
    cer_json_path = os.path.join(subdir_path, "cs.jsonl")
    if not os.path.isfile(cer_json_path):
        continue

    # with open(cer_json_path, "r") as f:
    #     samples = json.load(f)
    
    samples = []
    with open(cer_json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 跳过空行
                samples.append(json.loads(line.strip()))
    if len(samples) == 0:
        continue

    ref = samples[0]["ref"]
    positive = [x for x in samples if (x["cer"] == 0.0 and x["sim"] >= sim_threshold)]
    negative = [x for x in samples if x["cer"] >= cer_threshold]

    if len(positive) == 0 or len(negative) == 0:
        continue
    # 按 cer 降序排序负样本，选前x个
    negative_sorted = sorted(negative, key=lambda x: x["cer"], reverse=True)
    top_negs = negative_sorted[:max_neg_per_pos]

    for p in positive:
        for n in top_negs:
            all_pairs.append({
                "utt": f"{subdir}/{p['audio']}",
                "rejected_utt": f"{subdir}/{n['audio']}",
                "text": ref
            })

# 保存所有配对
output_path = os.path.join(parent_dir, "results/dpo_pairs.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_pairs, f, indent=2, ensure_ascii=False)

print(f"✅ 总共构造了 {len(all_pairs)} 个 DPO 数据对")