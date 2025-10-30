#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 src_dir/results 下读取一个 jsonl 文件，生成 Kaldi 风格数据文件。
jsonl 文件格式：
{"utt": "A3_0001/seed1", "rejected_utt": "A3_0001/seed2", "text": "今天的天气很好。"}
"""

import os
import json
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Kaldi-style data files from a single jsonl file under src_dir/results.")
    parser.add_argument("--src_dir", type=str, required=True, help="输入目录，需包含 results 子目录及 jsonl 文件")
    parser.add_argument("--des_dir", type=str, required=True, help="输出目录，用于存放 wav.scp/text/utt2spk/spk2utt")
    parser.add_argument("--spk_name", type=str, default="spk1", help="说话人名称")
    parser.add_argument("--json_name", type=str, default="cs_pairs.jsonl", help="results 目录下的 jsonl 文件名")
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = args.src_dir
    des_dir = args.des_dir
    json_name = args.json_name
    os.makedirs(des_dir, exist_ok=True)

    # === 查找 results 下的 jsonl 文件 ===
    jsonl_path = os.path.join(src_dir, "results", json_name)
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"❌ 未找到 jsonl 文件: {jsonl_path}")

    print(f"🔍 读取文件: {jsonl_path}")

    used_utts = set()
    utt2text = {}

    # === 读取 jsonl ===
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            for k in ["utt", "rejected_utt"]:
                if k in p and p[k]:
                    used_utts.add(p[k])
                    utt2text[p[k]] = p["text"]

    # === 写入 Kaldi 文件 ===
    wav_scp_path = os.path.join(des_dir, "wav.scp")
    text_path = os.path.join(des_dir, "text")
    utt2spk_path = os.path.join(des_dir, "utt2spk")
    spk2utt_path = os.path.join(des_dir, "spk2utt")

    with open(wav_scp_path, "w", encoding="utf-8") as fwav, \
         open(text_path, "w", encoding="utf-8") as ftext, \
         open(utt2spk_path, "w", encoding="utf-8") as fspk:

        for utt in sorted(used_utts):
            subdir, name = utt.split("/")
            wav_path = os.path.join(src_dir, subdir, name + ".wav")
            if not os.path.exists(wav_path):
                print(f"⚠️ 跳过缺失音频: {wav_path}")
                continue
            fwav.write(f"{utt} {os.path.abspath(wav_path)}\n")
            ftext.write(f"{utt} {utt2text[utt]}\n")
            fspk.write(f"{utt} {args.spk_name}\n")

    # === 构造 spk2utt ===
    spk2utt = defaultdict(list)
    for utt in used_utts:
        spk2utt[args.spk_name].append(utt)

    with open(spk2utt_path, "w", encoding="utf-8") as fspk2utt:
        for spk, utts in spk2utt.items():
            fspk2utt.write(f"{spk} {' '.join(sorted(utts))}\n")

    print(f"✅ 已完成：{src_dir} → {des_dir}，共 {len(used_utts)} 条样本")


if __name__ == "__main__":
    main()