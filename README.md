# VoiceTextBlender: Augmenting Large Language Models with Speech Capabilities via Single-Stage Joint Speech-Text Supervised Fine-Tuning


This repo contains the code for our NAACL 2025 paper: [VoiceTextBlender](https://arxiv.org/abs/2410.17485)

## Overview

We propose a novel single-stage joint speech-text SFT approach for training SpeechLMs using LoRA adapters.
Our model achieves excellent performance across various speech benchmarks while retaining performance on text-only benchmarks. 
Our 3B model even outperforms previous 7B or 13B SpeechLMs on most evaluated benchmarks.
Furthermore, our model exhibits emergent capabilities in handling previously unseen instructions and multi-turn mixed-modal conversations.

Specifically, we combine multi-turn text-only SFT data with single-turn speech-related SFT data during training.
To extend beyond speech-based QA tasks, we propose a novel data generation method that can create mixed-modal interleaving speech-text inputs.

## Installation

The code is based on an old version of NVIDIA [NeMo](https://github.com/NVIDIA/NeMo).

## Generating speech SFT data

## Running inference using a pre-trained model

## Training a new model



## Citation
```BibTeX
@inproceedings{vtblender,
    title={{VoiceTextBlender: Augmenting Large Language Models with Speech Capabilities via Single-Stage Joint Speech-Text Supervised Fine-Tuning}}, 
    author={Yifan Peng and Krishna C. Puvvada and Zhehuai Chen and Piotr Zelasko and He Huang and Kunal Dhawan and Ke Hu and Shinji Watanabe and Jagadeesh Balam and Boris Ginsburg},
    year={2025},
    booktitle={Proceedings of the 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL)},
}
```
