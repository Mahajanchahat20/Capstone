import os
import cv2
import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Model
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tqdm import tqdm
from google.colab import drive

# --- ANTI-FREEZE SETTINGS FOR COLAB ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
cv2.setNumThreads(0)

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
drive.mount('/content/drive')

BASE = "/content/drive/MyDrive/multimodal_project"
CSV_PATH = f"{BASE}/data.csv"
CLIPS_DIR = f"{BASE}/clips" 

MAX_LEN = 128
NUM_FRAMES = 16
IMAGE_SIZE = 112
AUDIO_SR = 16000 
AUDIO_DUR = 5    
BATCH_SIZE = 4   
EPOCHS = 15
LEARNING_RATE = 3e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATA PREPARATION
# ==========================================
def load_video_frames(video_path, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return torch.zeros((3, num_frames, image_size, image_size))

    step = max(total_frames // num_frames, 1)
    frame_indices = [i * step for i in range(num_frames)]
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            frames.append(frame)
            if len(frames) == num_frames: break
        count += 1
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))
        
    frames = np.array(frames)
    frames = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0
    
    normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    frames = torch.stack([normalize(f) for f in frames.permute(1, 0, 2, 3)], dim=0).permute(1, 0, 2, 3)
    return frames

def load_audio(video_path, target_sr=AUDIO_SR, max_duration=AUDIO_DUR):
    try:
        waveform, sr = torchaudio.load(video_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        max_len = target_sr * max_duration
        if waveform.size(1) > max_len:
            waveform = waveform[:, :max_len]
        elif waveform.size(1) < max_len:
            pad_len = max_len - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        return waveform.squeeze(0)
    except Exception:
        return torch.zeros(target_sr * max_duration)

class ARLDataset(Dataset):
    def __init__(self, df, clip_dir, tokenizer, max_len):
        self.df = df
        self.clip_dir = clip_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.clip_dir, f"{row['Video ID']}.mp4")
        
        video_tensor = load_video_frames(video_path)
        audio_tensor = load_audio(video_path)
        
        combined_text = f"{str(row['Hinglish Text'])} {self.tokenizer.sep_token} {str(row['Hindi Text'])}"
        encoding = self.tokenizer(
            combined_text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
                
        return {
            'video': video_tensor,
            'audio': audio_tensor,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(row['encoded_label'], dtype=torch.long)
        }

# ==========================================
# 3. TRUE PAPER IMPLEMENTATION: ARL MODEL
# ==========================================
class TrueARLModel(nn.Module):
    def __init__(self, num_classes):
        super(TrueARLModel, self).__init__()
        
        # --- 1. Encoders ---
        self.text_encoder = AutoModel.from_pretrained('xlm-roberta-base')
        self.video_encoder = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.video_encoder.fc = nn.Identity()
        self.audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        
        # Partial Fine-Tuning 
        for encoder in [self.text_encoder, self.video_encoder, self.audio_encoder]:
            for param in encoder.parameters(): param.requires_grad = False
                
        for name, param in self.text_encoder.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name: param.requires_grad = True
        for name, param in self.video_encoder.named_parameters():
            if "layer4" in name: param.requires_grad = True
        for name, param in self.audio_encoder.named_parameters():
            if "encoder.layers.11" in name: param.requires_grad = True
            
        self.shared_dim = 128
        self.text_proj = nn.Linear(768, self.shared_dim)
        self.video_proj = nn.Linear(512, self.shared_dim)
        self.audio_proj = nn.Linear(768, self.shared_dim)
        
        # --- 2. WEC (Weighted Encoder Calibration) ---
        self.wec_text = nn.Parameter(torch.tensor(1.0))
        self.wec_video = nn.Parameter(torch.tensor(1.0))
        self.wec_audio = nn.Parameter(torch.tensor(1.0))
        
        # Joint Classifier
        self.joint_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.shared_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # Hyperparameters from the Paper
        self.tau = 0.5  # Contribution threshold 
        self.alpha = 0.5 # Masking penalty (0.5 means 50% penalty to dominant modality)

    def forward(self, input_ids, attention_mask, video, audio, labels=None):
        # Feature Extraction
        with torch.no_grad(): 
            t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            v_out = self.video_encoder(video)
            a_out = self.audio_encoder(audio).last_hidden_state.mean(dim=1)
        
        # Projection & WEC Calibration
        t_feat = nn.functional.relu(self.text_proj(t_out)) * self.wec_text
        v_feat = nn.functional.relu(self.video_proj(v_out)) * self.wec_video
        a_feat = nn.functional.relu(self.audio_proj(a_out)) * self.wec_audio
        
        # --- 3. CISC (Contribution-Inverse Sample Calibration) ---
        if self.training and labels is not None:
            # We calculate the LOO masks without tracking gradients to save VRAM
            with torch.no_grad():
                f_all = torch.cat((t_feat, v_feat, a_feat), dim=1)
                f_no_t = torch.cat((torch.zeros_like(t_feat), v_feat, a_feat), dim=1)
                f_no_v = torch.cat((t_feat, torch.zeros_like(v_feat), a_feat), dim=1)
                f_no_a = torch.cat((t_feat, v_feat, torch.zeros_like(a_feat)), dim=1)
                
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                
                loss_all = loss_fn(self.joint_classifier(f_all), labels)
                loss_no_t = loss_fn(self.joint_classifier(f_no_t), labels)
                loss_no_v = loss_fn(self.joint_classifier(f_no_v), labels)
                loss_no_a = loss_fn(self.joint_classifier(f_no_a), labels)
                
                # Contribution calculation (Increase in loss when modality is removed)
                c_t = loss_no_t - loss_all
                c_v = loss_no_v - loss_all
                c_a = loss_no_a - loss_all
                
                # Apply Threshold (\tau) Masking
                mask_t = torch.where(c_t > self.tau, self.alpha, 1.0).unsqueeze(1)
                mask_v = torch.where(c_v > self.tau, self.alpha, 1.0).unsqueeze(1)
                mask_a = torch.where(c_a > self.tau, self.alpha, 1.0).unsqueeze(1)

            # Apply the calculated masks to the active gradient graph
            t_feat = t_feat * mask_t
            v_feat = v_feat * mask_v
            a_feat = a_feat * mask_a

        # 4. Final Late Fusion Prediction
        fused_features = torch.cat((t_feat, v_feat, a_feat), dim=1)
        logits = self.joint_classifier(fused_features)
        
        return logits

# ==========================================
# 4. TRAINING PIPELINE
# ==========================================
def main():
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    df['Label'] = df['Label'].astype(str).str.lower().str.strip()
    
    intent_mapping = {
        'inform': 'inform_explain', 'explain': 'inform_explain', 'emphasize': 'inform_explain', 'confirm': 'inform_explain',
        'complain': 'negative_conflict', 'criticize': 'negative_conflict', 'oppose': 'negative_conflict', 'refuse': 'negative_conflict', 'warn': 'negative_conflict',
        'joke': 'humor_mockery', 'taunt': 'humor_mockery',
        'advise': 'support', 'care': 'support',
        'doubt': 'doubt', 'other': 'other'
    }
    df['Label'] = df['Label'].map(intent_mapping).fillna('other')
    
    le = LabelEncoder()
    df['encoded_label'] = le.fit_transform(df['Label'])
    num_classes = len(le.classes_)
    
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['encoded_label'], random_state=42)
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    train_dataset = ARLDataset(train_df, CLIPS_DIR, tokenizer, MAX_LEN)
    val_dataset = ARLDataset(val_df, CLIPS_DIR, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TrueARLModel(num_classes).to(DEVICE)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['encoded_label']), y=train_df['encoded_label'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    best_model_path = f"{BASE}/neurips_arl_model.pth"

    print("Starting ARL Training (Trimodal WEC + CISC)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in train_pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            video = batch['video'].to(DEVICE)
            audio = batch['audio'].to(DEVICE) 
            labels = batch['label'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask, video, audio, labels=labels)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(correct/total)*100:.2f}%"})

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    batch['input_ids'].to(DEVICE), 
                    batch['attention_mask'].to(DEVICE), 
                    batch['video'].to(DEVICE), 
                    batch['audio'].to(DEVICE), 
                    labels=None
                )
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == batch['label'].to(DEVICE)).sum().item()
                val_total += batch['label'].size(0)

        epoch_val_acc = (val_correct/val_total)*100
        print(f"Epoch {epoch+1} Summary -> Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)

    print("\n================ FINAL EVALUATION ================")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Generating Report"):
            outputs = model(
                batch['input_ids'].to(DEVICE), 
                batch['attention_mask'].to(DEVICE), 
                batch['video'].to(DEVICE),
                batch['audio'].to(DEVICE)
            )
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
            
    print("\nClassification Report (Best Checkpoint):")
    print(classification_report(all_labels, all_preds, target_names=le.classes_, zero_division=0))

if __name__ == "__main__":
    main()
