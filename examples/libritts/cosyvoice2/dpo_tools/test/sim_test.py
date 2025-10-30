import os
import json
from tqdm import tqdm

# 配置参数
root_dir = "/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new-sft/data/male/test"  # 替换为你的目录
similarity_filename = "utt2similarity.json"
sim_threshold = 0.75

# 初始化统计变量
all_sims = []
pass_dirs = 0
total_dirs = 0
pass_ratio_1_0 = 0
pass_ratio_0_9 = 0
pass_ratio_0_7 = 0
per_dir_stats = []

# 遍历子目录
for subdir in tqdm(sorted(os.listdir(root_dir)), desc="统计子目录相似度"):
    full_path = os.path.join(root_dir, subdir)
    sim_path = os.path.join(full_path, similarity_filename)
    if not os.path.isfile(sim_path):
        continue

    with open(sim_path, "r", encoding="utf-8") as f:
        utt2sim = json.load(f)

    sims = [entry["sim"] for entry in utt2sim.values() if "sim" in entry]
    if not sims:
        continue

    total_dirs += 1
    all_sims.extend(sims)

    count_pass = sum(1 for s in sims if s >= sim_threshold)
    ratio = count_pass / len(sims)

    if ratio == 1.0:
        pass_ratio_1_0 += 1
    if ratio >= 0.9:
        pass_ratio_0_9 += 1
    if ratio >= 0.7:
        pass_ratio_0_7 += 1

    per_dir_stats.append({
        "folder": subdir,
        "pass_count": count_pass,
        "total": len(sims),
        "pass_ratio": ratio
    })

# 汇总
summary = {
    "total_dirs": total_dirs,
    "average_similarity": sum(all_sims) / len(all_sims) if all_sims else 0,
    f"dirs_with_all_sim>={sim_threshold}": pass_ratio_1_0,
    f"dirs_with_90%_sim>={sim_threshold}": pass_ratio_0_9,
    f"dirs_with_70%_sim>={sim_threshold}": pass_ratio_0_7,
    "details": per_dir_stats
}

# 保存与打印
os.makedirs(os.path.join(root_dir, "results"), exist_ok=True)
output_path = os.path.join(root_dir, "results/similarity_summary.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n📊 相似度统计完成：")
print(f"总目录数：{total_dirs}")
print(f"平均相似度：{summary['average_similarity']:.4f}")
print(f"SIM ≥ {sim_threshold:.2f} 的样本占比 = 100% 的目录数：{pass_ratio_1_0} ({pass_ratio_1_0 / total_dirs:.2%} if total_dirs else 0)")
print(f"SIM ≥ {sim_threshold:.2f} 的样本占比 ≥ 90% 的目录数：{pass_ratio_0_9} ({pass_ratio_0_9 / total_dirs:.2%} if total_dirs else 0)")
print(f"SIM ≥ {sim_threshold:.2f} 的样本占比 ≥ 70% 的目录数：{pass_ratio_0_7} ({pass_ratio_0_7 / total_dirs:.2%} if total_dirs else 0)")
print(f"📁 结果保存至：{output_path}")