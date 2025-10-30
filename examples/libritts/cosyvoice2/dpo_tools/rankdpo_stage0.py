# -*- coding: utf-8 -*-
"""
从 src_dir/results 下读取 RankDPO 三元组 jsonl（含 utt_group / rewards / text），
将 utt_group 中的所有 utt 写入 Kaldi 风格数据文件：wav.scp, text, utt2spk, spk2utt。

每行 JSONL 示例：
{
  "utt_group": ["text_38/sample_12", "text_38/sample_3", "text_38/sample_9"],
  "rewards": {"text_38/sample_12": 0.76219, "text_38/sample_3": 0.73146, "text_38/sample_9": 0.63971},
  "text": "司马懿他的妻子是一个普通官吏的女儿"
}
"""

import os
import json
import argparse
from collections import defaultdict, Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Kaldi-style data files from RankDPO triplet jsonl (src_dir/results/<json_name>)."
    )
    parser.add_argument("--src_dir", type=str, required=True,
                        help="输入目录，需包含 results 子目录及 <json_name> 文件")
    parser.add_argument("--des_dir", type=str, required=True,
                        help="输出目录，保存 wav.scp/text/utt2spk/spk2utt")
    parser.add_argument("--spk_name", type=str, default="spk1",
                        help="说话人名称（写入 utt2spk / spk2utt）")
    parser.add_argument("--json_name", type=str, default="css-train.jsonl",
                        help="results 目录下的 jsonl 文件名（默认 css.jsonl）")
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = os.path.abspath(args.src_dir)
    des_dir = os.path.abspath(args.des_dir)
    os.makedirs(des_dir, exist_ok=True)

    jsonl_path = os.path.join(src_dir, "results", args.json_name)
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"❌ 未找到 jsonl 文件: {jsonl_path}")

    print(f"🔍 读取 RankDPO 三元组文件: {jsonl_path}")

    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    # 收集所有 utt 与文本（每条三元组的三个 utt 共享同一 text）
    used_utts = set()
    utt2text = {}
    for p in pairs:
        utt_group = p.get("utt_group", [])
        txt = p.get("text", "")
        for u in utt_group:
            used_utts.add(u)
            # 若同一 utt 出现多次，以最后一次为准（通常文本一致）
            utt2text[u] = txt

    # 统计重复使用情况（同一 utt 出现在多个三元组中）
    all_utts = [u for p in pairs for u in p.get("utt_group", [])]
    cnt = Counter(all_utts)
    duplicate_utts = {u: c for u, c in cnt.items() if c > 1}
    print(f"📊 三元组数: {len(pairs)} | 去重后样本数: {len(used_utts)} | 重复使用样本数: {len(duplicate_utts)}")
    if duplicate_utts:
        print("🔁 重复样本（前 20 条）：")
        for u, c in list(duplicate_utts.items())[:20]:
            print(f" - {u}: {c} 次")

    # 写 Kaldi 文件
    wav_scp = os.path.join(des_dir, "wav.scp")
    text = os.path.join(des_dir, "text")
    utt2spk = os.path.join(des_dir, "utt2spk")
    spk2utt = os.path.join(des_dir, "spk2utt")

    missing_wav = 0
    with open(wav_scp, "w", encoding="utf-8") as fwav, \
         open(text, "w", encoding="utf-8") as ftext, \
         open(utt2spk, "w", encoding="utf-8") as fspk:
        for utt in sorted(used_utts):
            # 约定：utt 形如 "<subdir>/<name>"，音频位于 src_dir/<subdir>/<name>.wav
            try:
                subdir, name = utt.split("/", 1)
            except ValueError:
                print(f"⚠️ 跳过非法 utt（非 'a/b' 形式）: {utt}")
                continue
            wav_path = os.path.join(src_dir, subdir, name + ".wav")
            if not os.path.exists(wav_path):
                print(f"⚠️ 缺失音频，跳过: {wav_path}")
                missing_wav += 1
                continue
            fwav.write(f"{utt} {os.path.abspath(wav_path)}\n")
            ftext.write(f"{utt} {utt2text.get(utt, '')}\n")
            fspk.write(f"{utt} {args.spk_name}\n")

    # 写 spk2utt
    s2u = defaultdict(list)
    for utt in used_utts:
        s2u[args.spk_name].append(utt)
    with open(spk2utt, "w", encoding="utf-8") as f:
        for spk, utts in s2u.items():
            f.write(f"{spk} {' '.join(sorted(utts))}\n")

    print(f"✅ 已完成：{src_dir} → {des_dir}")
    if missing_wav:
        print(f"⚠️ 缺失音频数量: {missing_wav}")


if __name__ == "__main__":
    main()


