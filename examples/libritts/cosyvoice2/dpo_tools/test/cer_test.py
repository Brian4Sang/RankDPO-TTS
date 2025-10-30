import os
import json
import re
from jiwer import transforms
from tqdm import tqdm
import fastwer
import cn2an  # 新增：中文数字转阿拉伯数字模块

# =====================
# 配置路径
# =====================
input_txt = "/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/data/data/txt/long-test.txt"
audio_root = "/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-sft/data/female/test"

# =====================
# 加载参考文本（逐行对应）
# =====================
ref_texts = []
with open(input_txt, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            ref_texts.append(line)

# =====================
# 中文数值智能转换函数（如“六百一十九”→“619”）
# =====================
def normalize_number_cn(text):
    pattern = re.compile(r'[零〇一二三四五六七八九十百千万亿两]{2,}')
    def try_convert(match):
        try:
            return str(cn2an.cn2an(match.group(), "smart"))
        except:
            return match.group()  # 转换失败就保留原样
    return pattern.sub(try_convert, text)

# =====================
# 文本预处理函数
# =====================
to_lowercase = transforms.ToLowerCase()
remove_punctuation = transforms.RemovePunctuation()

def preprocess_text(text):
    text = to_lowercase(text)
    text = normalize_number_cn(text)  # 智能中文数字转阿拉伯数字
    text = remove_punctuation(text)
    text = text.replace(" ", "")
    return text

# =====================
# 每个子目录单独计算
# =====================
all_results = []
all_cers = []

for idx, ref_text in enumerate(tqdm(ref_texts, desc="逐目录计算 CER")):
    # idx = idx if idx < 418 else idx + 2
    
    folder_name = f"text_{idx}"
    subdir_path = os.path.join(audio_root, folder_name)

    if not os.path.isdir(subdir_path):
        print(f"⚠️ 子目录不存在: {subdir_path}")
        continue

    ref_text_processed = preprocess_text(ref_text)
    results = []

    for filename in os.listdir(subdir_path):
        if not filename.endswith(".lab"):
            continue

        audio_name = os.path.splitext(filename)[0]
        lab_path = os.path.join(subdir_path, filename)

        with open(lab_path, "r", encoding="utf-8") as f:
            hyp_text = f.read().strip()

        hyp_text_processed = preprocess_text(hyp_text)

        # 计算 CER
        cer = fastwer.score([hyp_text_processed], [ref_text_processed], char_level=True) / 100

        result_entry = {
            "audio": audio_name,
            "ref": ref_text,
            "hyp": hyp_text,
            "cer": cer
        }
        results.append(result_entry)
        all_results.append(result_entry)
        all_cers.append(cer)

    # 按 CER 排序
    results.sort(key=lambda x: x["cer"])

    # 保存当前子目录的 wer_results.json
    result_path = os.path.join(subdir_path, "cer_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ {folder_name} 已完成，共 {len(results)} 条")

# =====================
# 汇总整体 CER 统计
# =====================
if all_cers:
    avg_cer = sum(all_cers) / len(all_cers)
    zero_count = sum(1 for c in all_cers if c == 0)
    zero_ratio = zero_count / len(all_cers)

    summary = {
        "total_samples": len(all_cers),
        "average_cer": avg_cer,
        "zero_cer_count": zero_count,
        "zero_cer_ratio": zero_ratio,
        "top10_worst": sorted(all_results, key=lambda x: x["cer"], reverse=True)[:30]
    }

    os.makedirs(os.path.join(audio_root,"results"),exist_ok=True)
    summary_path = os.path.join(audio_root, "results/all_wer_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\n📊 汇总完成：总样本 = {len(all_cers)}，平均 CER = {avg_cer:.3f}")
    print(f"📊 CER = 0 的样本数 = {zero_count}（比例 = {zero_ratio:.2%}）")
    print(f"📊 汇总结果已保存到: {summary_path}")
else:
    print("❌ 没有成功评估任何样本")