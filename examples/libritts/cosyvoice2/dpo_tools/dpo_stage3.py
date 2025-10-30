#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从一个 pairs_dir（含 results/jsonl）和一个 token_dir（含 utt2speech_token.pt）
生成 utt2reject_speech_token.pt。
"""

import os
import json
import torch
from collections import OrderedDict
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate utt2reject_speech_token.pt from pair and token dirs.")
    parser.add_argument("--pairs_dir", type=str, required=True,
                        help="包含各子目录下 results/xxx.jsonl 的目录（如 outputs-new/dpo_data/zhihao/train）")
    parser.add_argument("--token_dir", type=str, required=True,
                        help="包含 utt2speech_token.pt 的目录")
    parser.add_argument("--json_name", type=str, default="cs_pairs.jsonl", help="jsonl文件名")
    parser.add_argument("--token_name", type=str, default="utt2speech_token.pt", help="token 文件名（默认 utt2speech_token.pt）")
    parser.add_argument("--output_name", type=str, default="utt2reject_speech_token.pt", help="输出文件名")
    return parser.parse_args()


def main():
    args = parse_args()

    pairs_dir = os.path.abspath(args.pairs_dir)
    token_dir = os.path.abspath(args.token_dir)

    token_path = os.path.join(token_dir, args.token_name)
    output_path = os.path.join(token_dir, args.output_name)

    if not os.path.exists(token_path):
        raise FileNotFoundError(f"❌未找到 token 文件: {token_path}")

    utt2tok = torch.load(token_path)
    print(f"加载 token: {token_path} ({len(utt2tok)} 条)")

    # === 找出 pairs_dir 下所有 jsonl 文件 ===
    jsonl_list = []
   
    results_path = os.path.join(pairs_dir, "results", args.json_name)
    if os.path.exists(results_path):
        jsonl_list.append(results_path)

    if not jsonl_list:
        raise FileNotFoundError(f"❌ 未找到任何 {args.json_name} 文件于 {results_path}")

    total_valid = total_missing = total_dup = 0
    utt2reject = OrderedDict()

    # === 处理每个 jsonl ===
    for jp in jsonl_list:
        with open(jp, "r", encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f if line.strip()]

        valid_pairs = missing = dup_cnt = 0

        for pair in pairs:
            utt = pair["utt"]
            rej = pair["rejected_utt"]

            if utt in utt2reject:
                dup_cnt += 1
                continue

            if rej in utt2tok:
                utt2reject[utt] = utt2tok[rej]
                valid_pairs += 1
            else:
                missing += 1
                print(f"⚠️缺失 token: {rej}")

        total_valid += valid_pairs
        total_missing += missing
        total_dup += dup_cnt
        print(f"处理 {jp} | 有效对: {valid_pairs} 缺失: {missing} 重复: {dup_cnt}")

    torch.save(dict(utt2reject), output_path)

    print(f"\n输出文件: {output_path}")
    print(f"总有效对: {total_valid}")
    print(f"⚠️缺失Token数  : {total_missing}")
    print(f"⚠️重复Token数  : {total_dup}")
    print("任务完成。")


if __name__ == "__main__":
    main()