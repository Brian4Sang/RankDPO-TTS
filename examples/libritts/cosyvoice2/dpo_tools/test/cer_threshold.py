import os
import json
from tqdm import tqdm

# =====================
# 配置路径和阈值参数
# =====================
audio_root = "/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new-sft/test/sft/female"
cer_threshold = 0.05  # CER 阈值
cer_strict_threshold = 0.05  # 新增：更严格的 CER 阈值

# =====================
# 初始化统计数据结构
# =====================
folder_stats = {
    "total_dirs": 0,
    "pass_ratio_1.0": 0,
    "pass_ratio_0.9": 0,
    "total_samples": 0,
    "strict_pass_count": 0,
    "details": []
}

# =====================
# 遍历所有子目录的 cer_results.json 文件
# =====================
for folder_name in tqdm(sorted(os.listdir(audio_root)), desc="检查各目录 CER"):
    subdir_path = os.path.join(audio_root, folder_name)
    if not os.path.isdir(subdir_path):
        continue

    result_path = os.path.join(subdir_path, "cer_results.json")
    if not os.path.isfile(result_path):
        continue

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取 {result_path}：{e}")
        continue

    cer_list = [r["cer"] for r in results if "cer" in r]
    if not cer_list:
        continue

    folder_stats["total_dirs"] += 1
    count_pass = sum(1 for c in cer_list if c < cer_threshold)
    count_strict_pass = sum(1 for c in cer_list if c < cer_strict_threshold)
    ratio = count_pass / len(cer_list)

    folder_stats["total_samples"] += len(cer_list)
    folder_stats["strict_pass_count"] += count_strict_pass

    if ratio >= 1.0:
        folder_stats["pass_ratio_1.0"] += 1
    if ratio >= 0.9:
        folder_stats["pass_ratio_0.9"] += 1

    folder_stats["details"].append({
        "folder": folder_name,
        "pass_count": count_pass,
        "total": len(cer_list),
        "pass_ratio": ratio
    })

# =====================
# 写入汇总结果
# =====================
summary_path = os.path.join(audio_root, "results/cer_threshold_summary.json")
os.makedirs(os.path.dirname(summary_path), exist_ok=True)

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(folder_stats, f, indent=4, ensure_ascii=False)

# =====================
# 打印简要统计信息
# =====================
print("\n📊 CER 阈值统计：")
print(f"总目录数：{folder_stats['total_dirs']}")
print(f"总样本数：{folder_stats['total_samples']}")
print(f"CER < {cer_threshold:.2f} 的语音占比 = 100% 的目录数：{folder_stats['pass_ratio_1.0']} ({folder_stats['pass_ratio_1.0'] / folder_stats['total_dirs']:.2%})")
print(f"CER < {cer_threshold:.2f} 的语音占比 ≥ 90% 的目录数：{folder_stats['pass_ratio_0.9']} ({folder_stats['pass_ratio_0.9'] / folder_stats['total_dirs']:.2%})")

if folder_stats['total_samples'] > 0:
    strict_ratio = folder_stats['strict_pass_count'] / folder_stats['total_samples']
else:
    strict_ratio = 0.0

print(f"CER < {cer_strict_threshold:.2f} 的总样本数：{folder_stats['strict_pass_count']} 占比：{strict_ratio:.2%}")
print(f"📁 详细信息保存至：{summary_path}")