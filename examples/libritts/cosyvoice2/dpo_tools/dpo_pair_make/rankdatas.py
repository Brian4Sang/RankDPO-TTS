import os
import json
from glob import glob
# Prepare rankpairs for train json

# === 设置路径 ===
triplet_path = "/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new/dpo_data/female/short/train-1200/results/css.jsonl"
score_root = "/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new/dpo_data/female/short/train-1200"
output_path = "/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new/dpo_data/female/short/train-1200/results/css-train.jsonl"

# === 1. 加载 triplets ===
with open(triplet_path, "r", encoding="utf-8") as f:
    triplets = [json.loads(line) for line in f if line.strip()]

# === 2. 构建 utt → score 映射 ===
utt2score = {}
for subdir in glob(os.path.join(score_root, "*")):
    score_file = os.path.join(subdir, "final_fuse.jsonl")
    if not os.path.exists(score_file):
        continue
    base = os.path.basename(subdir)
    
    with open(score_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            utt_id = f"{base}/{item['audio']}"
            utt2score[utt_id] = item.get("final_score")

# === 3. 过滤并保留每个正样本 reward 差最大的一组 ===
utt_best = {}
for t in triplets:
    u, m, r = t["utt"], t["mid_utt"], t["rej_utt"]
    if u not in utt2score or m not in utt2score or r not in utt2score:
        continue
    score_u = utt2score[u]
    score_r = utt2score[r]
    diff = score_u - score_r
    if u not in utt_best or diff > utt_best[u][1]:
        utt_best[u] = (t, diff)

# === 4. 组织输出结构 ===
final_data = []
for utt, (triplet, _) in utt_best.items():
    u, m, r = triplet["utt"], triplet["mid_utt"], triplet["rej_utt"]
    out = {
        "utt_group": [u, m, r],
        "rewards": {
            u: utt2score[u],
            m: utt2score[m],
            r: utt2score[r],
        },
        "text": triplet["text"]
    }
    final_data.append(out)

# === 5. 保存 ===
with open(output_path, "w", encoding="utf-8") as f:
    for item in final_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 构造完成，保存 {len(final_data)} 条样本到: {output_path}")