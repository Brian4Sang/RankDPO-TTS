import os
import json
import random
from glob import glob

# ===== 路径设置 =====
root_dir = "/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new/dpo_data/female/short/test-100"  # 每个子目录下有 final.jsonl
output_path = os.path.join(root_dir, "results","cf.jsonl")

# ===== 参数设置 =====
final_score_threshold = 0.775
chat_score_pos_threshold = 0.71

final_rej_score_threshold = 0.724
chat_score_neg_threshold = 0.68
max_pairs_per_group = 3

# ===== 开始构造 =====
all_pairs = []

for sub in sorted(glob(os.path.join(root_dir, "*"))):
    final_path = os.path.join(sub, "final_fuse.jsonl")
    if not os.path.exists(final_path):
        continue

    with open(final_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    base = os.path.basename(sub)

    pos_pool = [
        s for s in data
        if s.get("final_score", 0) >= final_score_threshold and s.get("chatscore", 0) >= chat_score_pos_threshold
        and "audio" in s and (s.get("reftext") or s.get("ref"))
    ]
    neg_pool = [
        s for s in data
        if s.get("final_score", 1) < final_rej_score_threshold and s.get("chatscore", 1) <= chat_score_neg_threshold
        and "audio" in s
    ]

    random.shuffle(pos_pool)
    random.shuffle(neg_pool)

    used_utts = set()
    pair_count = 0

    for p in pos_pool:
        if pair_count >= max_pairs_per_group:
            break
        u = f"{base}/{p['audio']}"
        if u in used_utts:
            continue

        for n in neg_pool:
            r = f"{base}/{n['audio']}"
            if r in used_utts or r == u:
                continue

            text = p.get("reftext") or p.get("ref") or ""
            if not text:
                continue

            all_pairs.append({
                "utt": u,
                "rejected_utt": r,
                "text": text
            })
            used_utts.update([u, r])
            pair_count += 1
            break

# ===== 写入输出 =====
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for item in all_pairs:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 构造完成，共生成 {len(all_pairs)} 条 DPO 数据对 -> {output_path}")