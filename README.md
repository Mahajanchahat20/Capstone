# Multimodal Intent Recognition in Dialogue (BEAR & ARL)

# Project Overview
This project implements and adapts two multimodal frameworks—BEAR and an ARL-based model—for intent classification on a custom dataset derived from a Hindi web series. The system leverages text (Hindi + Hinglish), audio, and video to perform multimodal understanding of conversational intent.

# Dataset (Custom Multimodal Dataset from Hindi Web Series)
This project uses a handmade multimodal dataset constructed from a Hindi web series, designed for intent classification in conversational settings.

# Overview
400 video clips extracted from a Hindi web series
1 CSV annotation file (data.csv)
Task: Intent Classification
Modalities: Text + Audio + Video
# Dataset Construction
Episodes from a Hindi web series were processed and segmented into short clips
Each clip corresponds to a single dialogue or conversational unit
Clips were manually aligned with:
Text (Hinglish + Hindi)
Intent label
Each sample therefore represents a complete multimodal instance of communication:
spoken dialogue + visual context + textual annotation + intent

# Data Format
Each row in data.csv corresponds to one sample:
Column	Description
Hinglish Text	     Conversational text in Hinglish (Hindi + English mix)
Hindi Text	       Hindi version/transcription of the dialogue
Video ID	         Identifier for the video clip
Label	             Annotated intent of the speaker

# Dataset Characteristics
Derived from real scripted dialogues
Captures:
Tone, sarcasm, humor, conflict, etc.
Naturally aligned modalities: Speech (audio),Facial/body expressions (video), Linguistic content (text)
However:
Moderate dataset size (400 samples)
Class imbalance present
Hinglish introduces linguistic variability
Label Settings Used

The dataset is used in two configurations:
1. Original Labels
Fine-grained intent categories
More challenging classification task
Preserves conversational nuance
2. Grouped Labels
Similar intents merged (e.g., inform + explain)
Reduces imbalance
Improves model stability

# Intents
Intent Classes
# Express Emotions / Attitudes
Complain
Praise
Apologise
Thank
Criticize
Care
Agree
Taunt
Flaunt
Oppose
Joke
Doubt
Acknowledge
Refuse
Warn
Emphasize
# Achieve Goals
Inform
Advise
Arrange
Introduce
Comfort
Leave
Prevent
Greet
Ask for Help
Ask for Opinions
Confirm
Explain
Invite
Plan
# Other
Other (Out-of-Scope / OOS)

# Multimodal Intent Classification using BEAR & ARL (Custom Dataset)

This repository contains implementations of two multimodal learning frameworks applied to a custom Hinglish multimodal dataset for intent classification.
# 1. BEAR Framework (CVPR 2025)
Paper: Uncertain Multimodal Intention and Emotion Understanding in the Wild

Key Ideas from Paper:
Multimodal learning under missing/uncertain modalities
Modality Asynchronous Prompt (MAP) for missing data
BEIFormer for reasoning between emotion and intention
Contrastive learning between multimodal features and label representations

# What Was Intended vs What Was Implemented

# BEAR (Intended)
The original BEAR framework:
Uses text + image/video + audio
Jointly models emotion and intention
Uses masked prompt-based reasoning (BEIFormer)
Exploits emotion–intention correlation
BEAR (Implemented in this repo)

This repository implements a BEAR-style adaptation:
# Implemented:
Trimodal pipeline (text + audio + video)
Feature extraction using:
XLM-Roberta (text)
Wav2Vec2 (audio)
VideoMAE (video)
Modality Asynchronous Prompt (MAP) for missing modalities
MulT-style cross-modal aggregation
Label-text contrastive reasoning
Feature caching for efficiency
# Differences from paper:
Intent-only classification (no emotion modeling)
Custom dataset instead of MINE dataset
No explicit BEIFormer masking with emotion-intention pairs
Practical engineering adaptation rather than exact reproduction

# Why These Adaptations Were Made

The original BEAR framework is designed for a large-scale setting where both emotion and intention are jointly modeled using rich multimodal data. In this project, the implementation is adapted to a custom dataset built from a Hindi web series, which differs in scale, annotations, and available modalities. These differences required several practical modifications.

The most significant change is the shift to intent-only classification. The BEAR paper relies on emotion–intention relationships, but this dataset contains only intent labels. Adding emotion labels would require extensive manual annotation or unreliable automatic methods, so the model is adapted to focus purely on intent. To retain some reasoning capability, intent labels are converted into text (e.g., “The intention is X”) and used in a contrastive learning setup, preserving the idea of aligning labels with multimodal features.

Another major difference is the use of a custom dataset instead of the MINE dataset. While MINE is large and diverse, this dataset is smaller (~400 samples) and derived from conversational clips of a Hindi web series. This makes it more realistic for dialogue understanding but introduces challenges like limited data and class imbalance. As a result, the implementation focuses more on robustness and adaptability rather than strict reproduction of the original setup.

The BEAR paper’s BEIFormer masking mechanism is also simplified. Since emotion labels are not available, full emotion–intention masking cannot be applied. Instead, a lighter version is used where label text representations are aligned with multimodal features through contrastive learning. This keeps the core idea of semantic reasoning without requiring dual-label supervision.

Feature extraction is handled in a practical way using pretrained models (XLM-R, Wav2Vec2, VideoMAE), with features cached to make training feasible on limited resources. Additionally, the model is restricted to a trimodal setup (text, audio, video), since these are the only modalities available in the dataset.

Finally, two label settings are used: original labels and grouped labels. The grouped version merges similar intent classes to reduce imbalance and improve training stability, while the original version preserves fine-grained distinctions for a more realistic evaluation.

Overall, these changes adapt BEAR from a large-scale research framework to a smaller, real-world multimodal dataset, while preserving its key ideas of multimodal fusion, missing modality handling, and label-aware reasoning.

# Citataion
```
@inproceedings{yang2025uncertain,
  title={Uncertain multimodal intention and emotion understanding in the wild},
  author={Yang, Qu and Shi, Qinghongya and Wang, Tongxin and Ye, Mang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24700--24709},
  year={2025}
}
```

# 2. ARL-Based Multimodal Model (NeurIPS 2025)

Paper: Adaptive Re-calibration Learning for Balanced Multimodal Intention Recognition

Key Ideas from Paper:
Multimodal fusion using pretrained encoders
Weighted Encoder Calibration (WEC) → learn modality importance
Contribution-Inverse Sample Calibration (CISC) → reduce dominant modality bias

# What Was Intended vs What Was Implemented

# ARL Model (Intended)
The ARL framework focuses on:
Multimodal feature fusion
Handling modality imbalance using:
WEC (learnable modality weights)
CISC (modality contribution-based masking)
ARL Model (Implemented in this repo)
# Implemented:
Trimodal encoding:
XLM-R (text)
R3D-18 (video)
Wav2Vec2 (audio)
Projection to shared feature space
WEC (Weighted Encoder Calibration)
CISC (Contribution-Inverse Sample Calibration)
Late fusion classification
# Important Implementation Detail:
Encoders are partially defined as trainable, but forward pass uses torch.no_grad()
This effectively freezes backbone encoders during training

# Why These Adaptations Were Made
The ARL framework is designed for multimodal intention recognition with balanced datasets and full end-to-end training. In this project, it is adapted to a custom dataset built from a Hindi web series, which differs in size, structure, and modality setup. These differences required practical modifications while keeping the core ideas of ARL intact.

The dataset consists of short video clips with Hinglish text, audio, and video, so the pipeline was adapted to load clips via Video ID, extract audio directly from video, and combine Hinglish and Hindi text into a single input. Pretrained models such as XLM-Roberta (text), Wav2Vec2 (audio), and R3D-18 (video) were used because they are well-suited for this type of data and are practical for implementation.

Another key change is that encoder forward passes are run under torch.no_grad(), effectively freezing them during training. This was done because the dataset is relatively small (~400 samples), and fully fine-tuning large models would lead to overfitting and unstable training. Instead, only projection layers, calibration parameters, and the classifier are trained, making the model more stable.

The implementation also includes label grouping, where similar intent classes are merged. This helps reduce class imbalance and improves learning stability, which is important for a small dataset with many fine-grained classes.

Despite these changes, the core contributions of the ARL paper are preserved. The model still applies Weighted Encoder Calibration (WEC) to learn modality importance and Contribution-Inverse Sample Calibration (CISC) to reduce the dominance of any single modality. These mechanisms are particularly useful in this dataset, where the importance of text, audio, and video can vary across samples.

Overall, the changes were made to adapt the ARL framework to a small, real-world multimodal dataset, ensuring the model remains both trainable and effective while preserving its key ideas.

# Citation
```
@inproceedings{yangadaptive,
  title={Adaptive Re-calibration Learning for Balanced Multimodal Intention Recognition},
  author={Yang, Qu and Li, Xiyang and Lin, Fu and Ye, Mang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
# How to Run

1. Install dependencies:
pip install transformers torchaudio torchvision pandas scikit-learn opencv-python

2. Update dataset path in code:
BASE = "your_path"

3. Run:
python bear_with_grouping.py
python arl_with_grouping.py
