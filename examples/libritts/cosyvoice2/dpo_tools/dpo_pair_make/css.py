#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 src_dir/results/final_fuse.jsonl 中筛选正负样本，构造 DPO 样本对。
每行格式示例：
{"audio": "sample_1", "ref": "今天的天气很好。", "chatscore": 0.74}
"""

import os
import json
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Construct cs_pairs.jsonl for DPO training from final_fuse.jsonl.")
    parser.add_argument("--src_dir", type=str, required=True,
                        help="输入目录，需包含 results/final_fuse.jsonl 和音频文件")
    parser.add_argument("--pos_thres", type=float, default=0.71,
                        help="正样本 chatscore 阈值 (>=)")
    parser.add_argument("--neg_thres", type=float, default=0.60,
                        help="负样本 chatscore 阈值 (<=)")
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="每个文本组最多生成的样本对数量")
    parser.add_argument("--json_name", type=str, default="final_fuse.jsonl",
                        help="输入 JSONL 文件名（默认 final_fuse.jsonl）")
    parser.add_argument("--output_name", type=str, default="cs_pairs.jsonl",
                        help="输出文件名（默认 cs_pairs.jsonl）")
    return parser.parse_args()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    src_dir = os.path.abspath(args.src_dir)
    json_path = os.path.join(src_dir, "results", args.json_name)
    out_path = os.path.join(src_dir, "results", args.output_name)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ 未找到输入文件: {json_path}")

    print(f"🔍 读取样本文件: {json_path}")
    samples = load_jsonl(json_path)

    pos_samples = [s for s in samples if s.get("chatscore", 0) >= args.pos_thres]
    neg_samples = [s for s in samples if s.get("chatscore", 0) <= args.neg_thres]

    print(f"📈 正样本数: {len(pos_samples)} | 📉 负样本数: {len(neg_samples)}")

    if not pos_samples or not neg_samples:
        print("⚠️ 正或负样本不足，跳过生成。")
        return

    random.shuffle(pos_samples)
    random.shuffle(neg_samples)

    used = set()
    pairs = []
    group_name = os.path.basename(src_dir)

    for pos in pos_samples:
        if pos["audio"] in used:
            continue
        for neg in neg_samples:
            if neg["audio"] in used or neg["audio"] == pos["audio"]:
                continue
            pairs.append({
                "utt": f"{group_name}/{pos['audio']}",
                "rejected_utt": f"{group_name}/{neg['audio']}",
                "text": pos.get("ref", "")
            })
            used.update([pos["audio"], neg["audio"]])
            if len(pairs) >= args.max_pairs:
                break
        if len(pairs) >= args.max_pairs:
            break

    save_jsonl(pairs, out_path)
    print(f"✅ 构造 {len(pairs)} 对样本 → {out_path}")


if __name__ == "__main__":
    main()