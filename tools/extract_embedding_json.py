#!/usr/bin/env python3
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import torch
from tqdm import tqdm
import torch.nn.functional as F
import json

def extract_embedding(wav_path):
    audio, sample_rate = torchaudio.load(wav_path)
    if isinstance(sample_rate,torch.Tensor):
        sample_rate = sample_rate.item()
    
    # sample_rate = int(sample_rate)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()})[0].flatten()
    return embedding

def compute_similarity(embedding, spk2embedding):
    if not isinstance(spk2embedding, dict):
        return None, None

    emb_tensor = torch.tensor(embedding)
    if emb_tensor.ndim > 1:
        emb_tensor = emb_tensor.squeeze()
    emb_tensor = F.normalize(emb_tensor, dim=0)

    best_spk = None
    best_sim = -1.0

    for spk, spk_emb in spk2embedding.items():
        emb_vec = spk_emb["embedding"] if isinstance(spk_emb, dict) else spk_emb
        spk_tensor = torch.tensor(emb_vec)
        if spk_tensor.ndim > 1:
            spk_tensor = spk_tensor.squeeze()
        spk_tensor = F.normalize(spk_tensor, dim=0)

        sim = torch.dot(emb_tensor, spk_tensor).item()
        if sim > best_sim:
            best_sim = sim
            best_spk = spk

    return best_spk, best_sim

def main(args):
    root_dir = args.root_dir
    spk2embedding = None
    if args.spk_emb_path and os.path.exists(args.spk_emb_path):
        spk2embedding = torch.load(args.spk_emb_path, weights_only=False)
        # spk2info = torch.load(args.spk_emb_path, weights_only=False)
        # spk2embedding = spk2info['zhihao']['embedding']

    all_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for subdir in tqdm(sorted(all_dirs), desc="处理每个子目录"):
        wav_files = [f for f in os.listdir(subdir) if f.endswith(".wav")]
        if not wav_files:
            continue

        utt2embedding = {}
        utt2sim = {}
        for wav_name in tqdm(wav_files, desc=f"提取: {os.path.basename(subdir)}", leave=False):
            wav_path = os.path.join(subdir, wav_name)
            utt_id = os.path.splitext(wav_name)[0]
            try:
                emb = extract_embedding(wav_path)
                utt2embedding[utt_id] = emb
                if spk2embedding:
                    best_spk, best_sim = compute_similarity(emb, spk2embedding)
                    utt2sim[utt_id] = {"spk": best_spk, "sim": best_sim}
            except Exception as e:
                print(f"❌ 处理失败: {wav_path} - {e}")

        # if utt2embedding:
        #     save_path = os.path.join(subdir, "utt2embedding.pt")
        #     torch.save(utt2embedding, save_path)
        #     print(f"✅ 保存: {save_path}")

        if utt2sim:
            sim_path = os.path.join(subdir, "utt2similarity.json")
            with open(sim_path, "w", encoding="utf-8") as f:
                json.dump(utt2sim, f, indent=2, ensure_ascii=False)
            print(f"📊 相似度保存: {sim_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="父目录，包含多个 text_x 子目录")
    parser.add_argument("--onnx_path", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--spk_emb_path", type=str, help="已有 spk2embedding.pt 文件路径")
    parser.add_argument("--num_thread", type=int, default=16)
    args = parser.parse_args()

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)

    main(args)