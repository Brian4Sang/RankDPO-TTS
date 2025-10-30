import sys
import os
import time
import torch
import torchaudio

sys.path.append('/brian/cosy/cosyvoice/new/cosyvoice/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2

# ========== 1) 加载模型 ==========
cosyvoice = CosyVoice2(
    '/brian/cosy/cosyvoice/CosyVoice/pretrained_models/CosyVoice2-0.5B',
    load_jit=True, load_trt=True, load_vllm=False, fp16=True
)

# ========== 2) 输出目录 ==========
output_root = '/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-sft/test/new/rank3-csc'
spk_a = 'tianqing'
spk_b = 'zhihao'
dir_a = os.path.join(output_root, spk_a)
dir_b = os.path.join(output_root, spk_b)
dir_merged = os.path.join(output_root, "merged")
os.makedirs(dir_a, exist_ok=True)
os.makedirs(dir_b, exist_ok=True)
os.makedirs(dir_merged, exist_ok=True)

dialogue = [
    ("tianqing", "你有没有过这种感觉？去参加老同学的婚礼，音乐一响，新人走出来，心里除了祝福，还有种复杂的感受，好像时间被按了快进键。"),
    ("zhihao", "我太懂了。看着台上的新人，脑子里却闪回过去的画面。昨天还在宿舍里打游戏、赶论文，今天就西装革履站在台上了。"),
    ("tianqing", "对！婚礼不仅是新人新篇章的开始，也会让台下的我们感受到强烈的时间跨越。那一刻真想说：时间过得太快了。"),
    ("zhihao", "没错，这种仪式感很容易触动人，让我们去审视自己走过的路，把过去和现在一下子连接起来。"),
    ("tianqing", "所以婚礼既是时间的印记，也是人生阶段的转换。它会触发很多关于个人成长和轨迹的思考。"),
    ("zhihao", "我觉得婚礼还是友谊的检验场和充电站。"),
    ("tianqing", "哦？怎么说？"),
    ("zhihao", "即使大家毕业后轨迹不同，有人在大城市打拼，有人回了老家，但朋友始终是生活里不可或缺的情感支持。"),
    ("tianqing", "对，就像那句话说的：不同的人走在不同的路上，可一旦见面，还是能聊得很开心。"),
    ("zhihao", "是啊，这种重聚提醒我们，生活再怎么变，友谊依然能跨越时间与空间。"),
    ("tianqing", "婚礼还会让人重新感知时间。随着年龄增长，大家都觉得时间流逝得更快。"),
    ("zhihao", "对，看到朋友们步入新阶段，肯定会让人反思自己的成长轨迹，比如学会了什么、性格和态度如何改变。"),
    ("tianqing", "这种反思不是焦虑，而是帮助我们认识自己，明白哪些经历真正塑造了我们。"),
    ("zhihao", "同时，婚礼也不只是两个人的结合，它还是家庭和社群的连接。"),
    ("tianqing", "没错，在快节奏生活中，婚礼就像一个暂停键，让我们有机会思考如何平衡工作、生活和人际关系。"),
    ("zhihao", "总结一下，婚礼带来的启发有好几层：提醒人生阶段转换、促进老友重聚、激发个人成长反思，还能帮我们放慢脚步，调整生活节奏。"),
]

# ========== 4) 合成并保存 ==========
sr = cosyvoice.sample_rate
gap_30ms = torch.zeros(1, int(0.03 * sr), dtype=torch.float32)
merged_pieces = []

for idx, (spk, text) in enumerate(dialogue):
    # 和你原来的写法一样：直接从生成器里取 seg 保存
    for seg_id, seg in enumerate(cosyvoice.inference_instruct_spk(text, spk, stream=False)):
        wav = seg['tts_speech']
        if spk == spk_a:
            out_path = os.path.join(dir_a, f"{idx:02d}_{spk}_{seg_id}.wav")
        else:
            out_path = os.path.join(dir_b, f"{idx:02d}_{spk}_{seg_id}.wav")
        torchaudio.save(out_path, wav, sr)
        print(f"Saved: {out_path}")

        # 合并对话：按顺序拼接，并在句子之间插 30ms
        merged_pieces.append(wav)
    if idx != len(dialogue) - 1:
        merged_pieces.append(gap_30ms)

# 合并完整对话音频
merged_wav = torch.cat(merged_pieces, dim=1)
ts = time.strftime("%Y%m%d_%H%M%S")
merged_path = os.path.join(dir_merged, f"dialogue_{ts}.wav")
torchaudio.save(merged_path, merged_wav, sr)
print(f"Saved merged dialogue: {merged_path}")

# 保存文本脚本
script_txt = os.path.join(dir_merged, f"dialogue_{ts}.txt")
with open(script_txt, "w", encoding="utf-8") as f:
    for i, (spk, text) in enumerate(dialogue):
        f.write(f"{i:02d} [{spk}] {text}\n")
print(f"Saved: {script_txt}")