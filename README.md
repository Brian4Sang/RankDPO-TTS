# Style-Aware RankDPO
Official implementation of the paper  
**“Style-Aware RankDPO for Stable Speech Synthesis” (ICASSP 2026, under review)** 

[Demos](https://fun-audio-llm.github.io); [Paper](https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf); [Modelscope](https://www.modelscope.cn/studios/iic/CosyVoice-300M), 

these framework is used for post-SFT fine-tuning and implemented upon:

**CosyVoice 2.0**: [Paper](https://arxiv.org/abs/2412.10117); [Modelscope](https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B); [HuggingFace](https://huggingface.co/spaces/FunAudioLLM/CosyVoice2-0.5B)

## Overview

This repository provides the implementation of **RankDPO**, a listwise preference optimization method for speech alignment,  
and **ChatScorer**, a learned reward model designed to evaluate **style consistency** and **natural conversational behavior** in TTS systems.

<p align="center">
  <img src="./asset/fig1_overall.png" width="700">
</p>

**Key ideas:**  
- RankDPO extends Direct Preference Optimization (DPO) to *listwise* training, allowing the model to leverage full ranking information across multiple candidate audios.  
- Meanwhile, ChatScorer predicts naturalness and conversational style while removing speaker bias via a GRL-based adversarial branch.

## Highlight🔥

-  **Listwise preference optimization (RankDPO)** — efficiently exploits ranking information from multiple samples per prompt.  
-  **ChatScorer reward model** — trained to measure *style stability* and *speaker-independent naturalness*.  
-  **Better style control** — reduces “machine-like” or unstable utterances in conversational TTS.  
-  **Modular training framework** — supports DPO, RankDPO, and DiffRO variants under a unified pipeline.  

## Install

### Clone and install

- Clone the repo
    ``` sh
    git clone --recursive https://github.com/Brian4Sang/RankDPO-TTS.git
    cd RankDPO-TTS
    git submodule update --init --recursive
    ```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

    ``` sh
    conda create -n rankdpo -y python=3.10
    conda activate rankdpo
    # pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platforms.
    conda install -y -c conda-forge pynini==2.1.5
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
    
    # If you encounter sox compatibility issues
    # ubuntu
    sudo apt-get install sox libsox-dev
    # centos
    sudo yum install sox sox-devel
    ```

### Cosyvoice2 Model download

We strongly recommend that you download their pretrained `CosyVoice2-0.5B` model and `CosyVoice-ttsfrd` resource.

``` python
# SDK模型下载
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

``` sh
# git模型下载，请确保已安装git lfs
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

Optionally, you can unzip `ttsfrd` resource and install `ttsfrd` package for better text normalization performance.

Notice that this step is not necessary. If you do not install `ttsfrd` package, we will use WeTextProcessing by default.

``` sh
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

### Basic Usage

Follow the code below for detailed usage of each model.

``` python
to be continued ...
```

## Results

to  be continued ...
<img src="./asset/fig3_chat_eval.png" width="250px">


## Evaluation Metrics

### Pass@K (Capability Upper Bound)
**Pass@K** evaluates whether the model retains *any* valid solution path within its stochastic generation space.  

This metric reflects the **upper-bound potential** of the model by checking the *existence* of at least one correct response.  
A reduction in Pass@K indicates **mode collapse**, meaning that low-probability but high-quality solution modes have disappeared from the model's output space.

---

### Failure@K (Stability Lower Bound)
**Failure@K** measures the model's *worst-case reliability* by requiring all \( K \) generations to be correct.  


This metric captures the **lower-bound performance** of the model, reflecting its **single-shot stability** and consistency under repeated generation.

---

- **Pass@K** assesses whether the model *can* succeed (capability upper bound).  
- **Failure@K** assesses whether the model *always* succeeds (stability lower bound).


## Acknowledge

1. We borrowed a lot of code from [Cosyvoice].

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.
