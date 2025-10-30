#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from glob import glob
from pathlib import Path
import csv

def load_scores_from_jsonl(path, score_key="score"):
    scores = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if score_key in obj:
                try:
                    v = float(obj[score_key])
                    scores.append(v)
                except (TypeError, ValueError):
                    pass
    return scores

def main():
    parser = argparse.ArgumentParser(description="统计多个子目录下 jsonl 的打分分布")
    parser.add_argument("root_dir", type=str, nargs="?", default="/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new-sft/test/sft/male", help="根目录（默认当前目录），递归查找 *.jsonl")
    parser.add_argument("--score-key", type=str, default="score", help="分数字段名（默认: score）")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    jsonl_paths = [p for p in glob(str(root_dir / "**" / "chatScore.jsonl"), recursive=True)]
    if not jsonl_paths:
        print("未找到任何 .jsonl 文件。")
        return

    all_scores = []
    files_all_gt_065 = 0
    files_with_scores = 0
    per_file_rows = []

    for path in jsonl_paths:
        scores = load_scores_from_jsonl(path, score_key=args.score_key)
        if len(scores) == 0:
            per_file_rows.append([path, 0, "", "", "", "", "", ""])
            continue

        files_with_scores += 1
        all_scores.extend(scores)

        n = len(scores)
        mean_v = sum(scores) / n
        c_lt_05   = sum(1 for s in scores if s < 0.5)
        c_05_06   = sum(1 for s in scores if 0.5 <= s < 0.6)
        c_06_07   = sum(1 for s in scores if 0.6 <= s < 0.7)
        c_ge_07   = sum(1 for s in scores if s >= 0.7)
        all_gt_065_flag = int(all(s > 0.65 for s in scores))
        files_all_gt_065 += all_gt_065_flag

        per_file_rows.append([
            path, n, f"{mean_v:.5f}",
            c_lt_05, c_05_06, c_06_07, c_ge_07,
            all_gt_065_flag
        ])

    N = len(all_scores)
    if N == 0:
        print("所有 .jsonl 都没有可用分数。")
        return

    overall_mean = sum(all_scores) / N
    b_lt_05   = sum(1 for s in all_scores if s < 0.5)
    b_05_06   = sum(1 for s in all_scores if 0.5 <= s < 0.6)
    b_06_07   = sum(1 for s in all_scores if 0.6 <= s < 0.7)
    b_ge_07   = sum(1 for s in all_scores if s >= 0.7)

    file_ratio_all_gt_065 = (files_all_gt_065 / files_with_scores) if files_with_scores > 0 else 0.0

    # ===== 保存结果到 root_dir/results =====
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 每文件统计 CSV
    out_csv = results_dir / "chatscore_score.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "n_scores", "mean", "count_<0.5", "count_[0.5,0.6)", "count_[0.6,0.7)", "count_>=0.7", "all_scores_>0.65"])
        w.writerows(per_file_rows)

    # 总体统计 JSONL
    out_jsonl = results_dir / "chatscore_summary.jsonl"
    stats = {
        "total_scores": N,
        "mean": round(overall_mean, 5),
        "count_<0.5": b_lt_05,
        "pct_<0.5": round(b_lt_05 / N, 4),
        "count_[0.5,0.6)": b_05_06,
        "pct_[0.5,0.6)": round(b_05_06 / N, 4),
        "count_[0.6,0.7)": b_06_07,
        "pct_[0.6,0.7)": round(b_06_07 / N, 4),
        "count_>=0.7": b_ge_07,
        "pct_>=0.7": round(b_ge_07 / N, 4),
        "jsonl_files_total": len(jsonl_paths),
        "jsonl_files_with_scores": files_with_scores,
        "files_all_scores_gt_0.65": files_all_gt_065,
        "ratio_files_all_scores_gt_0.65": round(file_ratio_all_gt_065, 4)
    }
    with open(out_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(stats, ensure_ascii=False) + "\n")

    print("====== 总体统计已完成 ======")
    print(f"- 文件级统计: {out_csv.resolve()}")
    print(f"- 总体统计:   {out_jsonl.resolve()}")

if __name__ == "__main__":
    main()