import os
import json
import subprocess
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import shutil
import sys

def debug_print(message, file=sys.stderr):
    print(message, file=file)
    file.flush()

def resample_audio(input_path, output_path, target_sr=16000):
    """
    安全地重采样音频文件
    
    :param input_path: 输入音频文件路径
    :param output_path: 输出音频文件路径
    :param target_sr: 目标采样率
    :return: 是否成功
    """
    try:
        # 读取音频文件
        data, orig_sr = sf.read(input_path)
        
        # 处理多声道音频
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        # 如果采样率不同，重采样
        if orig_sr != target_sr:
            data = librosa.resample(
                data, 
                orig_sr=orig_sr, 
                target_sr=target_sr
            )
        
        # 写入新文件
        sf.write(output_path, data, target_sr)
        return True
    except Exception as e:
        debug_print(f"重采样 {input_path} 失败: {str(e)}")
        return False

# ==== 配置 ====
input_audio_dir = "/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new/test-final/true-final/tianqing"
output_json_path = "/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new/test-final/true-final/tianqing/dns_scores.json"
dnsmos_script_path = "/brian/cosy/DNS-Challenge/DNSMOS/dnsmos_local.py"
batch_size = 500
temp_audio_root = "temp_audio_batches"

# 获取所有音频文件
def get_audio_files(directory):
    supported_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_extensions:
                audio_paths.append(os.path.join(root, file))
    return audio_paths

# 清理和准备临时目录
if os.path.exists(temp_audio_root):
    shutil.rmtree(temp_audio_root)
os.makedirs(temp_audio_root, exist_ok=True)

# 确保输出目录存在
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

# 获取音频文件列表
audio_paths = get_audio_files(input_audio_dir)

# 检查是否有音频文件
if not audio_paths:
    debug_print("❌ 未找到任何音频文件")
    exit(1)

debug_print(f"🔍 找到 {len(audio_paths)} 条音频文件")

# 分批处理
all_results = []

for batch_id, start in tqdm(
    enumerate(range(0, len(audio_paths), batch_size)),
    total=(len(audio_paths) + batch_size - 1) // batch_size,
    desc="📦 分批处理"):

    end = start + batch_size
    sub_paths = audio_paths[start:end]
    batch_dir = os.path.join(temp_audio_root, f"batch_{batch_id}")
    os.makedirs(batch_dir, exist_ok=True)

    path_map = {}  # 映射
    processed_paths = []  # 成功处理的音频路径

    for idx, origin in enumerate(sub_paths):
        linkname = f"sample_{idx}.wav"
        linkpath = os.path.join(batch_dir, linkname)
        
        # 使用新的重采样函数
        if resample_audio(origin, linkpath):
            path_map[os.path.realpath(linkpath)] = os.path.realpath(origin)
            processed_paths.append(linkpath)
        else:
            debug_print(f"❌ 处理音频 {origin} 失败")

    # 如果没有成功处理的音频，跳过本批次
    if not processed_paths:
        debug_print(f"❌ 批次 {batch_id} 没有可处理的音频")
        continue

    batch_csv = os.path.join(batch_dir, "mos_results.csv")

    debug_print(f"🚀 处理 batch {batch_id} ({start} ~ {end})")
    debug_print(f"  批次目录: {batch_dir}")
    debug_print(f"  批次CSV: {batch_csv}")
    
    try:
        # 运行DNSMOS脚本
        result = subprocess.run([
            "python", dnsmos_script_path,
            "-t", batch_dir,
            "-o", batch_csv,
        ], check=True, capture_output=True, text=True)
        
        debug_print("DNSMOS脚本标准输出:")
        debug_print(result.stdout)
        debug_print("DNSMOS脚本标准错误:")
        debug_print(result.stderr)
    
    except subprocess.CalledProcessError as e:
        debug_print(f"❌ 批次 {batch_id} 处理失败:")
        debug_print("标准输出: " + e.stdout)
        debug_print("标准错误: " + e.stderr)
        continue

    try:
        # 检查CSV文件
        if not os.path.exists(batch_csv):
            debug_print(f"❌ CSV文件 {batch_csv} 不存在")
            continue
        
        if os.path.getsize(batch_csv) == 0:
            debug_print(f"❌ CSV文件 {batch_csv} 为空")
            continue

        df = pd.read_csv(batch_csv)
        debug_print("CSV文件内容:")
        debug_print(df)
        
        # 处理列名问题
        if 'filename' not in df.columns:
            debug_print("❌ CSV文件列名异常")
            debug_print("现有列名: " + str(list(df.columns)))
            
            if 'Unnamed: 0' in df.columns:
                df = df.rename(columns={'Unnamed: 0': 'filename'})
            else:
                continue

        # 替换路径
        df["filename"] = df["filename"].map(lambda x: path_map.get(os.path.realpath(x), x))
        all_results.append(df)
    
    except Exception as e:
        debug_print(f"❌ 批次 {batch_id} CSV处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# 合并结果
if not all_results:
    debug_print("❌ 没有处理成功的结果")
    exit(1)

df_all = pd.concat(all_results, ignore_index=True)

# 生成结果字典
results_dict = {}
for _, row in df_all.iterrows():
    filepath = row["filename"]
    results_dict[filepath] = {
        "dnsmos": round(row["P808_MOS"], 3),
        "sig": round(row["SIG"], 3),
        "bak": round(row["BAK"], 3),
        "ovrl": round(row["OVRL"], 3),
    }

# 写入JSON文件
with open(output_json_path, "w", encoding="utf-8") as fout:
    json.dump(results_dict, fout, ensure_ascii=False, indent=2)

print(f"✅ 所有音频处理完成，打分结果保存至 {output_json_path}")
print(f"📄 共处理 {len(audio_paths)} 条音频")

# 打印统计信息
if results_dict:
    mos_scores = [item['dnsmos'] for item in results_dict.values()]
    print("\nMOS分数统计:")
    print(f"最小值: {min(mos_scores)}")
    print(f"最大值: {max(mos_scores)}")
    print(f"平均值: {np.mean(mos_scores):.3f}")
    print(f"中位数: {np.median(mos_scores):.3f}")
