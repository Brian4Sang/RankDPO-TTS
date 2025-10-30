
---

## Overview

This repository provides the implementation of **RankDPO**, a listwise preference optimization method for speech alignment,  
and **ChatScorer**, a learned reward model designed to evaluate **style consistency** and **natural conversational behavior** in TTS systems.

<p align="center">
  <img src="docs/fig1_overall.png" width="700">
</p>

**Key idea:**  
RankDPO extends Direct Preference Optimization (DPO) to *listwise* training, allowing the model to leverage full ranking information across multiple candidate audios.  
Meanwhile, ChatScorer predicts naturalness and conversational style while removing speaker bias via a GRL-based adversarial branch.

---

## Highlights

-  **Listwise preference optimization (RankDPO)** — efficiently exploits ranking information from multiple samples per prompt.  
-  **ChatScorer reward model** — trained to measure *style stability* and *speaker-independent naturalness*.  
-  **Better style control** — reduces “machine-like” or unstable utterances in conversational TTS.  
-  **Modular training framework** — supports DPO, RankDPO, and DiffRO variants under a unified pipeline.  

---

## Installation

### Clone the repository
```bash
git clone https://github.com/Brian4Sang/RankDPO-TTS.git
cd RankDPO-ChatScorer
