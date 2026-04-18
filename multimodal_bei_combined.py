# ============================================================
# BEAR-STYLE INTENT-ONLY IMPLEMENTATION
# ============================================================

# -----------------------------
# 1. INSTALL + IMPORTS
# -----------------------------
!pip install -q transformers torchaudio torchvision pandas scikit-learn tqdm opencv-python accelerate sentencepiece

import os
import cv2
import math
import json
import torch
import torchaudio
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer,
    AutoModel,
    VideoMAEImageProcessor,
    VideoMAEModel
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -----------------------------
# 2. CONFIG
# -----------------------------
BASE = "/content/drive/MyDrive/multimodal_project"
CSV_PATH = f"{BASE}/data.csv"
CLIPS_DIR = f"{BASE}/clips"
CACHE_DIR = f"{BASE}/bear_cache_intent_only"
os.makedirs(CACHE_DIR, exist_ok=True)

TEXT_MODEL_NAME = "xlm-roberta-base"
VIDEO_MODEL_NAME = "MCG-NJU/videomae-base"

AUDIO_SR = 16000
MAX_AUDIO_SECONDS = 8
NUM_VIDEO_FRAMES = 16
TARGET_DIM = 768

BATCH_SIZE = 8
EPOCHS = 25
LR = 5e-5
TEMP = 0.05
CONTRASTIVE_WEIGHT = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42

USE_LABEL_GROUPING = True

# -----------------------------
# 3. LOAD DATA
# -----------------------------
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv(CSV_PATH)
df["Label"] = df["Label"].astype(str).str.strip().str.lower()

for col in ["Hinglish Text", "Hindi Text", "Video ID"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column in CSV: {col}")

# -----------------------------
# 4. LABEL GROUPING
# -----------------------------
if USE_LABEL_GROUPING:
    mapping = {
        "other": "other", "care": "other", "comfort": "other",
        "praise": "other", "apologise": "other", "thank": "other",
        "greet": "other",

        "joke": "joke",
        "taunt": "taunt",

        "inform": "inform", "explain": "inform",
        "emphasize": "inform", "confirm": "inform",

        "complain": "complain", "criticize": "complain",

        "oppose": "oppose", "doubt": "oppose", "refuse": "oppose",

        "advise": "advise", "plan": "advise",
        "prevent": "advise", "invite": "advise",

        "acknowledge": "acknowledge", "agree": "acknowledge",

        "ask for opinions": "ask for opinions",
        "ask for help": "ask for opinions"
    }
    df["Label"] = df["Label"].replace(mapping)

print(df["Label"].value_counts())

labels = sorted(df["Label"].unique())
label2id = {lab: i for i, lab in enumerate(labels)}
id2label = {i: lab for lab, i in label2id.items()}

print(label2id)

# -----------------------------
# 5. TRAIN / TEST SPLIT
# -----------------------------
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["Label"],
    random_state=RANDOM_STATE
)

# -----------------------------
# 6. TEXT ENCODER
# -----------------------------
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(device).eval()

def combine_text(hinglish, hindi):
    hinglish = "" if pd.isna(hinglish) else str(hinglish).strip()
    hindi = "" if pd.isna(hindi) else str(hindi).strip()

    if hinglish and hindi:
        return f"Hinglish: {hinglish} [SEP] Hindi: {hindi}"
    elif hinglish:
        return hinglish
    elif hindi:
        return hindi
    else:
        return ""

def encode_texts(text_list, max_length=128):
    enc = text_tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = text_encoder(**enc).last_hidden_state[:, 0]

    return out.detach().cpu()

# -----------------------------
# 7. AUDIO ENCODER
# -----------------------------
wav2vec = torchaudio.pipelines.WAV2VEC2_BASE.get_model().to(device).eval()

def load_audio_from_video(path, target_sr=AUDIO_SR, max_seconds=MAX_AUDIO_SECONDS):
    try:
        wav, sr = torchaudio.load(path)

        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

        max_len = int(target_sr * max_seconds)
        if wav.shape[1] > max_len:
            wav = wav[:, :max_len]
        else:
            wav = F.pad(wav, (0, max_len - wav.shape[1]))

        return wav
    except:
        return None

def encode_audio(path):
    wav = load_audio_from_video(path)
    if wav is None:
        return torch.zeros(TARGET_DIM), 1

    wav = wav.to(device)
    with torch.no_grad():
        feats, _ = wav2vec.extract_features(wav)
        feat = feats[-1].mean(dim=1).squeeze(0).cpu()

    return feat, 0

# -----------------------------
# 8. VIDEO ENCODER
# -----------------------------
video_processor = VideoMAEImageProcessor.from_pretrained(VIDEO_MODEL_NAME)
video_encoder = VideoMAEModel.from_pretrained(VIDEO_MODEL_NAME).to(device).eval()

def sample_video_frames(path, num_frames=NUM_VIDEO_FRAMES):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    idxs = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames

def encode_video(path):
    frames = sample_video_frames(path)
    if frames is None:
        return torch.zeros(TARGET_DIM), 1

    inputs = video_processor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        out = video_encoder(pixel_values=pixel_values)
        feat = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()

    return feat, 0

# -----------------------------
# 9. FEATURE CACHE
# -----------------------------
def build_label_text(label_name):
    return f"The intention is {label_name}."

def extract_and_cache(split_df, split_name):
    out_path = f"{CACHE_DIR}/{split_name}.pt"
    if os.path.exists(out_path):
        return out_path

    data = []
    all_texts = [
        combine_text(row["Hinglish Text"], row["Hindi Text"])
        for _, row in split_df.iterrows()
    ]

    batch_text_feats = []
    for i in range(0, len(all_texts), 32):
        feats = encode_texts(all_texts[i:i+32], max_length=128)
        batch_text_feats.append(feats)
    batch_text_feats = torch.cat(batch_text_feats, dim=0)

    for idx, (_, row) in enumerate(split_df.iterrows()):
        vid = str(row["Video ID"]).strip()
        video_path = f"{CLIPS_DIR}/{vid}.mp4"

        text_feat = batch_text_feats[idx]
        text_missing = 0 if all_texts[idx] != "" else 1

        audio_feat, audio_missing = encode_audio(video_path)
        video_feat, video_missing = encode_video(video_path)

        data.append({
            "text_feat": text_feat,
            "audio_feat": audio_feat,
            "video_feat": video_feat,
            "text_missing": text_missing,
            "audio_missing": audio_missing,
            "video_missing": video_missing,
            "label_id": label2id[row["Label"]],
            "label_text": build_label_text(row["Label"])
        })

    torch.save(data, out_path)
    return out_path

train_cache = extract_and_cache(train_df, "train")
test_cache = extract_and_cache(test_df, "test")

# -----------------------------
# 10. DATASET
# -----------------------------
class BEARDataset(torch.utils.data.Dataset):
    def __init__(self, cache_path):
        self.data = torch.load(cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {
            "text_feat": x["text_feat"],
            "audio_feat": x["audio_feat"],
            "video_feat": x["video_feat"],
            "text_missing": torch.tensor(x["text_missing"], dtype=torch.long),
            "audio_missing": torch.tensor(x["audio_missing"], dtype=torch.long),
            "video_missing": torch.tensor(x["video_missing"], dtype=torch.long),
            "label_id": torch.tensor(x["label_id"], dtype=torch.long),
            "label_text": x["label_text"]
        }

train_ds = BEARDataset(train_cache)
test_ds = BEARDataset(test_cache)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# 11. MODEL (MAP + MulT + BEIFormer)
# -----------------------------
class ModalityAsynchronousPrompt(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.text_prompt = nn.Parameter(torch.randn(dim))
        self.audio_prompt = nn.Parameter(torch.randn(dim))
        self.video_prompt = nn.Parameter(torch.randn(dim))

    def forward(self, t, a, v, t_missing, a_missing, v_missing):
        tp = self.text_prompt.unsqueeze(0).expand_as(t)
        ap = self.audio_prompt.unsqueeze(0).expand_as(a)
        vp = self.video_prompt.unsqueeze(0).expand_as(v)

        t = t + t_missing.unsqueeze(1).float() * tp
        a = a + a_missing.unsqueeze(1).float() * ap
        v = v + v_missing.unsqueeze(1).float() * vp

        return t, a, v

class PairCrossAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        out, _ = self.attn(q, kv, kv)
        return self.norm(q + out)

class MulTStyleAggregator(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.t_from_a = PairCrossAttention(dim)
        self.t_from_v = PairCrossAttention(dim)
        self.a_from_t = PairCrossAttention(dim)
        self.a_from_v = PairCrossAttention(dim)
        self.v_from_t = PairCrossAttention(dim)
        self.v_from_a = PairCrossAttention(dim)

    def forward(self, t, a, v):
        t = t.unsqueeze(1)
        a = a.unsqueeze(1)
        v = v.unsqueeze(1)

        ta = self.t_from_a(t, a)
        tv = self.t_from_v(t, v)
        at = self.a_from_t(a, t)
        av = self.a_from_v(a, v)
        vt = self.v_from_t(v, t)
        va = self.v_from_a(v, a)

        return torch.cat([t, a, v, ta, tv, at, av, vt, va], dim=1)

class BEIFormer(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, f_prom, f_ma):
        q = f_prom.unsqueeze(1)
        out, _ = self.cross_attn(q, f_ma, f_ma)
        out = self.norm1(q + out)
        out = self.norm2(out + self.ffn(out))
        return out.squeeze(1)

class BEARIntentModel(nn.Module):
    def __init__(self, num_classes, dim=768):
        super().__init__()
        self.map_module = ModalityAsynchronousPrompt(dim)
        self.mult_agg = MulTStyleAggregator(dim)
        self.bei = BEIFormer(dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(dim, num_classes)

    def encode_prompt(self, prompt_texts):
        return encode_texts(prompt_texts, max_length=64).to(next(self.parameters()).device)

    def encode_gt_text(self, gt_texts):
        return encode_texts(gt_texts, max_length=32).to(next(self.parameters()).device)

    def forward(self, t, a, v, tm, am, vm, label_texts=None):
        t, a, v = self.map_module(t, a, v, tm, am, vm)
        f_ma = self.mult_agg(t, a, v)

        f_prom = self.encode_prompt(["The data contains [MASK] intention."] * t.size(0))
        f_beif = self.bei(f_prom, f_ma)

        logits = self.classifier(self.dropout(f_beif))

        f_gt = self.encode_gt_text(label_texts) if label_texts else None
        return logits, f_beif, f_gt

# -----------------------------
# 12. LOSS + TRAINING
# -----------------------------
def nce_loss(x, y, tau=TEMP):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    logits = torch.matmul(x, y.T) / tau
    targets = torch.arange(x.size(0), device=x.device)
    return F.cross_entropy(logits, targets)

weights = torch.tensor(
    [sum(Counter(train_df["Label"]).values()) / Counter(train_df["Label"])[lab] for lab in sorted(label2id.keys())],
    dtype=torch.float32
).to(device)

ce_loss = nn.CrossEntropyLoss(weight=weights)

model = BEARIntentModel(len(label2id)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# 13. TRAINING
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        t = batch["text_feat"].to(device)
        a = batch["audio_feat"].to(device)
        v = batch["video_feat"].to(device)

        tm = batch["text_missing"].to(device)
        am = batch["audio_missing"].to(device)
        vm = batch["video_missing"].to(device)

        y = batch["label_id"].to(device)
        label_texts = batch["label_text"]

        logits, f_beif, f_gt = model(t, a, v, tm, am, vm, label_texts)

        loss = ce_loss(logits, y) + CONTRASTIVE_WEIGHT * nce_loss(f_beif, f_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: {total_loss:.4f}")

# -----------------------------
# 14. TESTING
# -----------------------------
model.eval()
preds, true = [], []

with torch.no_grad():
    for batch in test_loader:
        t = batch["text_feat"].to(device)
        a = batch["audio_feat"].to(device)
        v = batch["video_feat"].to(device)

        tm = batch["text_missing"].to(device)
        am = batch["audio_missing"].to(device)
        vm = batch["video_missing"].to(device)

        y = batch["label_id"].to(device)

        logits, _, _ = model(t, a, v, tm, am, vm)
        preds.extend(logits.argmax(1).cpu().tolist())
        true.extend(y.cpu().tolist())

print(classification_report(true, preds))
