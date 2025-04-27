# VoiceTextBlender

This repo contains the code for our paper:

Yifan Peng*, Krishna C. Puvvada*, Zhehuai Chen*, Piotr Zelasko, He Huang, Kunal Dhawan, Ke Hu, Shinji Watanabe, Jagadeesh Balam, and Boris Ginsburg, "VoiceTextBlender: Augmenting Large Language Models with Speech Capabilities via Single-Stage Joint Speech-Text Supervised Fine-Tuning," in Proc. NAACL, 2025. [[arXiv](https://arxiv.org/abs/2410.17485)]

## Overview

Recent studies have augmented large language models (LLMs) with speech understanding capabilities, leading to SpeechLMs. However, existing SpeechLMs typically require multi-stage curriculum learning and suffer from catastrophic forgetting of the original text-only capabilities.
In this work, we propose a novel training pipeline, which streamlines the training process, enhances speech understanding performance, and maintains text-only performance of SpeechLMs.
Specifically, we combine multi-turn text-only supervised fine-tuning (SFT) data with three types of single-turn speech-related SFT data and perform single-stage training.

## Demonstrations

Please find various demonstrations in our paper or the `imgs` directory in this repo.

## Installation

The code is based on an old version of NVIDIA [NeMo](https://github.com/NVIDIA/NeMo).
We use a docker container to train and decode the model. Please build the container and install relevant packages.

## Data Generation Scripts

As described in our paper, we combine four types of data to train VoiceTextBlender:
- Text-only conversations
- ASR and AST
- SQA generated from ASR data
- Mixed-modal SFT generated with TTS

Our training data is prepared in [Lhotse's format in NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html).
The first two types of data are commonly used in LLM or SpeechLM training. Please refer to our paper for data sources and detailed statistics of these data.
The last two types of data are newly generated. Below are the data generation scripts.

### SQA generated from ASR data

Given a transcript from an ASR dataset, we prompt an LLM to generate a pair of question and answer.
This is implemented in a python script: [`generate_audioqa_from_llm.py`](vtblender_scripts/generate_audioqa_from_llm.py)

We submit parallel jobs to a SLURM GPU cluster using `submitit`. 
The script can be customized for other environments, data, or LLMs.

### Mixed-modal SFT generated with TTS

We create mixed-modal SFT data from single-turn text SFT data. We randomly select consecutive sentences in the user turn and synthesize speech using a pre-trained TTS model. During training, these sentences will be replaced by the synthesized speech, leading to mixed-modal user input.
The script is [`tts_generate_mixedmodal.py`](vtblender_scripts/tts_generate_mixedmodal.py), which first downloads a text SFT dataset from Hugging Face and then performs TTS. Similarly, we submit parallel jobs on a SLURM cluster, but the script can be customized to other environments.

## Training Scripts

The training python script is [`examples/multimodal/speech_llm/modular_audio_gpt_train.py`](examples/multimodal/speech_llm/modular_audio_gpt_train.py)

The training config file is [`singlestage_gemma_enc-ft_adp-ft_llm-lora_lr1e-4_max100k.yaml`](examples/multimodal/speech_llm/conf/full_canary/singlestage_gemma_enc-ft_adp-ft_llm-lora_lr1e-4_max100k.yaml)

We launch the training job on a SLURM cluster using [`train_N8.sh`](vtblender_scripts/train_N8.sh).

## Inference Scripts

The inference script is [`examples/multimodal/speech_llm/modular_audio_gpt_eval.py`](examples/multimodal/speech_llm/modular_audio_gpt_eval.py)

We provide an example launching script for SQA: [`test_sqa.sh`](vtblender_scripts/test_sqa.sh)

## Citation
```BibTeX
@inproceedings{vtblender,
    title={{VoiceTextBlender}: Augmenting Large Language Models with Speech Capabilities via Single-Stage Joint Speech-Text Supervised Fine-Tuning}, 
    author={Yifan Peng and Krishna C. Puvvada and Zhehuai Chen and Piotr Zelasko and He Huang and Kunal Dhawan and Ke Hu and Shinji Watanabe and Jagadeesh Balam and Boris Ginsburg},
    year={2025},
    booktitle={Proceedings of the 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL)},
}
```
