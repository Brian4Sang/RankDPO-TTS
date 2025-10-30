#!/usr/bin/env python3
import os
import time
import json
import argparse
import multiprocessing

import torch
import pandas as pd
from tqdm import tqdm

def job_rankdpo(
    utt_list,
    parquet_file,
    utt2parquet_file,
    spk2parquet_file,
    utt2wav,
    utt2text,
    utt2spk,
    utt2embedding,
    spk2embedding,
    utt2rank  # ← 新：以 best_utt 为键，值含 {'utts','speech_tokens','rewards'}
):
    start_time = time.time()

    # 仅记录真正写入的数据，避免 df 与映射不对齐
    kept_utts = []

    data_list = []
    wav_list, text_list, spk_list = [], [], []
    uttembedding_list, spkembedding_list = [], []

    # 任意 rank-K 的列
    rank_utts_list = []
    rank_tokens_list = []
    rank_rewards_list = []

    for best_utt in tqdm(utt_list, desc=f"Processing {os.path.basename(parquet_file)}"):
        # 基础校验（best_utt 必须在基础映射中）
        if best_utt not in utt2wav or best_utt not in utt2text or best_utt not in utt2spk:
            continue
        try:
            audio_bytes = open(utt2wav[best_utt], 'rb').read()
        except Exception:
            continue

        # 读取该 best_utt 对应的一整组（K 可变）
        if best_utt not in utt2rank:
            continue
        group = utt2rank[best_utt]

        # 组字段：必须都有，且长度一致
        ut_group = group.get("utts", None)
        tok_group = group.get("speech_tokens", None)
        rew_group = group.get("rewards", None)
        if not isinstance(ut_group, list) or not isinstance(tok_group, list) or not isinstance(rew_group, list):
            continue
        if not (len(ut_group) == len(tok_group) == len(rew_group)) or len(ut_group) < 3:
            continue

        # 推进元数据（基于 best_utt）
        kept_utts.append(best_utt)
        data_list.append(audio_bytes)
        wav_list.append(utt2wav[best_utt])
        text_list.append(utt2text[best_utt])
        spk_list.append(utt2spk[best_utt])
        uttembedding_list.append(utt2embedding[best_utt])
        spkembedding_list.append(spk2embedding[utt2spk[best_utt]])

        # 推进 rank-K 列
        rank_utts_list.append(ut_group)
        rank_tokens_list.append(tok_group)
        rank_rewards_list.append(rew_group)

    # === 构建 DataFrame（严格按 kept_utts 对齐） ===
    df = pd.DataFrame()
    df['utt'] = kept_utts                       # best_utt
    df['wav'] = wav_list
    df['audio_data'] = data_list
    df['text'] = text_list
    df['spk'] = spk_list
    df['utt_embedding'] = uttembedding_list
    df['spk_embedding'] = spkembedding_list

    # 任意 rank-K 的三列
    df['rank_utts'] = rank_utts_list
    df['rank_speech_tokens'] = rank_tokens_list
    df['rank_rewards'] = rank_rewards_list

    # 写 parquet
    df.to_parquet(parquet_file)

    # 写映射（只包含真正写入的 best_utt 与出现过的 spk）
    with open(utt2parquet_file, 'w', encoding='utf-8') as f:
        json.dump({k: parquet_file for k in kept_utts}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, 'w', encoding='utf-8') as f:
        json.dump({k: parquet_file for k in sorted(set(spk_list))}, f, ensure_ascii=False, indent=2)

    print(f"[{os.path.basename(parquet_file)}] done in {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet', type=int, default=1000)
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--src_dir', type=str, required=True, help='source dir with wav.scp/text/utt2spk/embeddings')
    parser.add_argument('--des_dir', type=str, required=True, help='output directory')
    parser.add_argument('--rank_pt_name', type=str, default='utt2rank_speech_token.pt',
                        help='filename of the unified rank-K pt (under --src_dir)')
    args = parser.parse_args()

    os.makedirs(args.des_dir, exist_ok=True)

    # 读基础映射
    utt2wav, utt2text, utt2spk = {}, {}, {}
    with open(os.path.join(args.src_dir, 'wav.scp'), encoding='utf-8') as f:
        for l in f:
            l = l.strip().split()
            utt2wav[l[0]] = l[1]
    with open(os.path.join(args.src_dir, 'text'), encoding='utf-8') as f:
        for l in f:
            l = l.strip().split()
            utt2text[l[0]] = ' '.join(l[1:])
    with open(os.path.join(args.src_dir, 'utt2spk'), encoding='utf-8') as f:
        for l in f:
            l = l.strip().split()
            utt2spk[l[0]] = l[1]

    utt2embedding = torch.load(os.path.join(args.src_dir, 'utt2embedding.pt'))
    spk2embedding = torch.load(os.path.join(args.src_dir, 'spk2embedding.pt'))

    # 读统一 Rank-K 结构的 pt
    rank_pt_path = os.path.join(args.src_dir, args.rank_pt_name)
    utt2rank = torch.load(rank_pt_path)

    # best_utt 列表（键）
    utts = list(utt2rank.keys())

    # 多进程切分
    pool = multiprocessing.Pool(processes=args.num_processes)
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []

    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, f'parquet_{i:09d}.tar')
        utt2parquet_file = os.path.join(args.des_dir, f'utt2parquet_{i:09d}.json')
        spk2parquet_file = os.path.join(args.des_dir, f'spk2parquet_{i:09d}.json')

        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)

        pool.apply_async(
            job_rankdpo,
            (
                utts[j: j + args.num_utts_per_parquet],
                parquet_file,
                utt2parquet_file,
                spk2parquet_file,
                utt2wav,
                utt2text,
                utt2spk,
                utt2embedding,
                spk2embedding,
                utt2rank
            )
        )

    pool.close()
    pool.join()

    # 汇总列表
    with open(os.path.join(args.des_dir, 'data.list'), 'w', encoding='utf-8') as f1, \
         open(os.path.join(args.des_dir, 'utt2data.list'), 'w', encoding='utf-8') as f2, \
         open(os.path.join(args.des_dir, 'spk2data.list'), 'w', encoding='utf-8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')

    print("RankDPO parquet 构建完成。")

if __name__ == '__main__':
    main()