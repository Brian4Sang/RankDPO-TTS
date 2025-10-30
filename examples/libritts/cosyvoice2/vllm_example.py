import sys
import os
import argparse
import torchaudio
from tqdm import tqdm

sys.path.append('/brian/cosy/cosyvoice/CosyVoice/third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="CosyVoice2 批量生成语音样本")
parser.add_argument('--input_txt', type=str, default="/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/data/data/txt/test.txt",
                    required=False, help="输入文本路径，每行一条语句")
parser.add_argument('--output_dir', type=str, default="/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs/test/male/A-ori-zero",
                    required=False, help="输出音频保存根目录")
parser.add_argument('--start_line', type=int, default=0, help="从第几行开始处理")
parser.add_argument('--num_samples', type=int, default=10, help="每条文本生成多少个样本")
args = parser.parse_args()

input_txt = args.input_txt
output_base_dir = args.output_dir
start_line = args.start_line
num_samples_per_text = args.num_samples
prompt_texts = "听众朋友大家好，我是张志浩，欢迎收听本期的张志浩讲历史"

os.makedirs(output_base_dir, exist_ok=True)

# -------------------------------
# 初始化模型
# -------------------------------
cosyvoice = CosyVoice2(
    '/brian/cosy/cosyvoice/CosyVoice/pretrained_models/CosyVoice2-0.5B',
    load_jit=True, load_trt=True, load_vllm=True, fp16=True
)

# -------------------------------
# 读取输入文本并逐条合成
# -------------------------------
fails = 0
with open(input_txt, "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines[start_line:], start=start_line):
    text = line.strip()
    if not text:
        continue  # 跳过空行

    output_dir = os.path.join(output_base_dir, f'text_{idx}')
    os.makedirs(output_dir, exist_ok=True)
    
    prompt_speech_16k = load_wav('/brian/cosy/cosyvoice/CosyVoice/asset/zhihao.wav', 16000)

    try:
        for sample_num in range(1, num_samples_per_text + 1):
            set_all_random_seed(sample_num)

            for seg_id, seg in enumerate(cosyvoice.inference_zero_shot(text, prompt_texts, prompt_speech_16k,stream=False)):
                filename = os.path.join(output_dir, f'sample_{sample_num}.wav')
                torchaudio.save(filename, seg['tts_speech'], cosyvoice.sample_rate)
                print(f'Saved: {filename}')
    except Exception as e:
        print(f"❌ 第 {idx} 行文本处理失败：{e}")
        fails += 1

print("✅ 所有处理完成，失败总数：", fails)

    


