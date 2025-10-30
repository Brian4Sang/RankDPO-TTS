#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
import argparse

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
    utt2triplet
):
    start_time = time.time()
    data_list = []
    wav_list, text_list, spk_list = [], [], []
    uttembedding_list, spkembedding_list = [], []
    speech_token_list, mid_token_list, rej_token_list = [], [], []
    rewards_list = []

    for utt in tqdm(utt_list, desc=f"Processing {os.path.basename(parquet_file)}"):
        if utt not in utt2wav or utt not in utt2text or utt not in utt2spk:
            continue
        try:
            audio_bytes = open(utt2wav[utt], 'rb').read()
        except:
            continue

        triplet = utt2triplet[utt]
        data_list.append(audio_bytes)
        wav_list.append(utt2wav[utt])
        text_list.append(utt2text[utt])
        spk_list.append(utt2spk[utt])
        uttembedding_list.append(utt2embedding[utt])
        spkembedding_list.append(spk2embedding[utt2spk[utt]])

        speech_token_list.append(triplet["speech_token"])
        mid_token_list.append(triplet["mid_utt"]["speech_token"])
        rej_token_list.append(triplet["rej_utt"]["speech_token"])

        rewards_list.append([
            triplet["reward"],
            triplet["mid_utt"]["reward"],
            triplet["rej_utt"]["reward"]
        ])

    df = pd.DataFrame()
    df['utt'] = utt_list
    df['wav'] = wav_list
    df['audio_data'] = data_list
    df['text'] = text_list
    df['spk'] = spk_list
    df['utt_embedding'] = uttembedding_list
    df['spk_embedding'] = spkembedding_list
    
    df['speech_token'] = speech_token_list
    df['mid_speech_token'] = mid_token_list
    df['rej_speech_token'] = rej_token_list
    df['rewards'] = rewards_list

    df.to_parquet(parquet_file)

    with open(utt2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in set(spk_list)}, f, ensure_ascii=False, indent=2)

    print(f"[{os.path.basename(parquet_file)}] done in {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet', type=int, default=1000)
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--src_dir', type=str, required=True, help='source dir with wav.scp/text/utt2spk/embeddings')
    parser.add_argument('--des_dir', type=str, required=True, help='output directory')
    args = parser.parse_args()

    os.makedirs(args.des_dir, exist_ok=True)

    utt2wav, utt2text, utt2spk = {}, {}, {}
    with open(f'{args.src_dir}/wav.scp') as f:
        for l in f:
            l = l.strip().split()
            utt2wav[l[0]] = l[1]
    with open(f'{args.src_dir}/text') as f:
        for l in f:
            l = l.strip().split()
            utt2text[l[0]] = ' '.join(l[1:])
    with open(f'{args.src_dir}/utt2spk') as f:
        for l in f:
            l = l.strip().split()
            utt2spk[l[0]] = l[1]

    utt2embedding = torch.load(f'{args.src_dir}/utt2embedding.pt')
    spk2embedding = torch.load(f'{args.src_dir}/spk2embedding.pt')
    triplet_pt_path = os.path.join(args.src_dir,"utt2mid2rej_speech_token.pt")
    utt2triplet = torch.load(str(triplet_pt_path))

    utts = list(utt2triplet.keys())

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
                utt2triplet
            )
        )

    pool.close()
    pool.join()

    with open(f'{args.des_dir}/data.list', 'w') as f1, \
         open(f'{args.des_dir}/utt2data.list', 'w') as f2, \
         open(f'{args.des_dir}/spk2data.list', 'w') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')

    print("RankDPO parquet 构建完成。")


if __name__ == "__main__":
    main()