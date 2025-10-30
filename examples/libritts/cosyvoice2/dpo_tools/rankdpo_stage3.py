#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 pairs_dir（含 results/jsonl）与 token_dir（含 utt2speech_token.pt）
生成 RankDPO 用的 utt2mid2rej_speech_token.pt 文件。

输入 JSONL 格式示例：
{
    "utt_group": ["A3_0001/seed1", "A3_0001/seed2", "A3_0001/seed3"],
    "rewards": {
        "A3_0001/seed1": 0.92,
        "A3_0001/seed2": 0.75,
        "A3_0001/seed3": 0.21
    }
}
"""

import os
import json
import torch
from collections import OrderedDict
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate utt2mid2rej_speech_token.pt from triplet jsonl and token files.")
    parser.add_argument("--pairs_dir", type=str, required=True,
                        help="包含 results/jsonl 的目录（如 outputs-new/dpo_data/zhihao/train）")
    parser.add_argument("--token_dir", type=str, required=True,
                        help="包含 utt2speech_token.pt 的目录")
    parser.add_argument("--json_name", type=str, default="css-train.jsonl", help="results 目录下的 jsonl 文件名")
    parser.add_argument("--token_name", type=str, default="utt2speech_token.pt", help="输入 token 文件名")
    parser.add_argument("--output_name", type=str, default="utt2mid2rej_speech_token.pt", help="输出文件名")
    return parser.parse_args()


def main():
    args = parse_args()

    pairs_dir = os.path.abspath(args.pairs_dir)
    token_dir = os.path.abspath(args.token_dir)

    json_path = os.path.join(pairs_dir, "results", args.json_name)
    token_path = os.path.join(token_dir, args.token_name)
    output_path = os.path.join(token_dir, args.output_name)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ 未找到 triplet 文件: {json_path}")
    if not os.path.exists(token_path):
        raise FileNotFoundError(f"❌ 未找到 token 文件: {token_path}")

    print(f"🔍 加载 triplet: {json_path}")
    print(f"🔍 加载 token:   {token_path}")

    triplets = [json.loads(line) for line in open(json_path, encoding="utf-8") if line.strip()]
    utt2token = torch.load(token_path)

    utt2triplet = OrderedDict()
    missing = 0

    for t in triplets:
        utts = t["utt_group"]
        rewards = t["rewards"]

        try:
            entry = {
                "speech_token": utt2token[utts[0]],
                "reward": rewards[utts[0]],
                "mid_utt": {
                    "speech_token": utt2token[utts[1]],
                    "reward": rewards[utts[1]],
                },
                "rej_utt": {
                    "speech_token": utt2token[utts[2]],
                    "reward": rewards[utts[2]],
                },
            }
            utt2triplet[utts[0]] = entry
        except KeyError as e:
            print(f"⚠️ 缺失 token: {e}")
            missing += 1

    torch.save(utt2triplet, output_path)

    print(f"\n✅ 输出文件: {output_path}")
    print(f"📦 有效样本数: {len(utt2triplet)}")
    print(f"⚠️ 缺失 token 数: {missing}")
    print("任务完成。")


if __name__ == "__main__":
    main()