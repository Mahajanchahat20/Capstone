# Multimodal Intent Classification using BEAR & ARL
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
