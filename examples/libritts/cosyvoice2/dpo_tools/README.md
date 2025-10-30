# DPO 数据准备说明

## dpo_stage 0  
脚本：`dpo_tools/dpo_stage0.py`  
**功能：**  
从 `src_dir` 下的results子目录中读取指定的 jsonl 文件（需要根据筛选规则提前生成），提取其中的 `utt` 与 `rejected_utt` 字段，生成合并后的 Kaldi 风格数据文件(stage0)

- jsonl 文件格式示例：

  {"utt": "A3_0001/seed1", "rejected_utt": "A3_0001/seed2", "text": "今天的天气很好。"}
  ```

### 输入要求  
- `--src_dir`：包含若干子目录的输入目录，子目录中results必须有一个 jsonl 文件（默认名 `cs_pairs.jsonl`），结构示例：
  ```
src_dir/
  ├── results/
  │     └── cs_pairs.jsonl
  ├── A3_0001/
  │     ├── seed1.wav
  │     └── seed2.wav
  ├── A3_0002/
  │     ├── seed1.wav
  │     └── seed2.wav
  └── ...
  ```

### 输出内容  
在 `des_dir` 下生成四个文件：
```
wav.scp
text
utt2spk
spk2utt
```
---

## dpo_stage3
脚本：`dpo_tools/dpo_stage3.py`  
**功能：**  
根据一个 `pairs_dir`（包含结果【同上的cs_pairs. jsonl】文件）与一个 `token_dir`（包含utt2speech_token.pt文件），匹配 `utt` 与其对应的拒绝样本 token，生成 `utt2reject_speech_token.pt`。

### 输入要求  
- `--pairs_dir`：包含结果 jsonl 的目录（默认名 `cs_pairs.jsonl`），通常结构为：
  ```
  pairs_dir/
    └── results/
          └── cs_pairs.jsonl
  ```
- `--token_dir`：包含 `utt2speech_token.pt` 的目录  
  ```
  token_dir/
    └── utt2speech_token.pt
  ```


### 输出内容  
在 `token_dir` 下生成：
```
utt2reject_speech_token.pt
```
其中每个 `utt` 对应其拒绝样本的语音 token，用于后续 DPO 训练过滤或配对。




# RankDPO 数据准备说明

## 脚本：`dpo_tools/rank3dpo_stage0.py`  
**功能：**  
从 `src_dir/results` 下读取 RankDPO 三元组 jsonl 文件（含 `utt`、`mid_utt`、`rej_utt`），  
并生成 Kaldi 风格的数据文件：`wav.scp`、`text`、`utt2spk`、`spk2utt`。

---

### 输入目录要求  
`--src_dir` 需包含以下结构：  
```
src_dir/
  ├── results/
  │     └── css-train.jsonl
  ├── A3_0001/
  │     ├── seed1.wav
  │     ├── seed2.wav
  │     └── seed3.wav
  ├── A3_0002/
  │     ├── seed1.wav
  │     ├── seed2.wav
  │     └── seed3.wav
  └── ...
```

`css-train.jsonl` 文件格式示例：
```json
{
  "utt_group": ["A3_0001/seed1", "A3_0001/seed2", "A3_0001/seed3"],
  "rewards": {
    "A3_0001/seed1": 0.92,
    "A3_0001/seed2": 0.75,
    "A3_0001/seed3": 0.21
  },
  "text": "司马懿他的妻子是一个普通官吏的女儿"
}
```

---
### 输出内容  
在 `des_dir` 下生成以下文件：
```
wav.scp
text
utt2spk
spk2utt
```
每条样本对应一条语音路径和文本，供后续 RankDPO 训练使用。

# RankDPO Token 匹配生成说明

## 脚本：`dpo_tools/dpo_stage3_rank.py`  
**功能：**  
根据一个 RankDPO 三元组结果文件（`results/jsonl`）与语音 token 文件（`utt2speech_token.pt`），  
生成三元组对应的 token+reward 文件 `utt2mid2rej_speech_token.pt`。

---

### 输入目录要求  

- `--pairs_dir`：包含三元组结果的目录  
  ```
  pairs_dir/
    └── results/
          └── css-train.jsonl
  ```
  ```
  token_dir/
    └── utt2speech_token.pt
  ```

`css-train.jsonl` 文件格式示例：
```json
{
  "utt_group": ["A3_0001/seed1", "A3_0001/seed2", "A3_0001/seed3"],
  "rewards": {
    "A3_0001/seed1": 0.92,
    "A3_0001/seed2": 0.75,
    "A3_0001/seed3": 0.21
  },
  "text": "司马懿他的妻子是一个普通官吏的女儿"
}
```

---

### 输出内容  

在 `token_dir` 下生成：
```
utt2mid2rej_speech_token.pt
```

该文件为一个字典结构：  
每个主 `utt` 包含自身及其中等样本（mid_utt）与拒绝样本（rej_utt）的 token 与 reward 信息，  
用于 RankDPO 模型训练阶段进行三元组式偏好对齐。