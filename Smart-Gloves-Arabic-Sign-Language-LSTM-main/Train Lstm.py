import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import json
from collections import defaultdict

warnings.filterwarnings('ignore')

# Configuration

DATA_FILE = "sign_languagedataset.csv"

MODEL_PATH   = "lstm_sign_language_model.pth"
CLASSES_PATH = "classes.npy"
INFO_PATH    = "training_info.json"
SCALER_PATH  = "scaler.pkl"

BATCH_SIZE    = 16
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
NUM_EPOCHS    = 100
LEARNING_RATE = 0.0005
DROPOUT       = 0.2
WEIGHT_DECAY  = 1e-5
SEQUENCE_LENGTH = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Load Dataset

print("\n" + "="*60)
print("Loading dataset...")
print("="*60)

df = pd.read_csv(DATA_FILE)

print(f"\nTotal samples  : {len(df):,}")
print(f"Total gestures : {df['Gesture_ID'].nunique()}")
print(f"Labels         : {df['Label'].nunique()}")

# Features

feature_cols = [
    'flex_thumb', 'flex_index', 'flex_middle', 'flex_ring', 'flex_pinky',
    'ax', 'ay', 'az',
    'gx', 'gy', 'gz',
    'pitch', 'roll'
]

flex_cols = ['flex_thumb','flex_index', 'flex_middle', 'flex_ring', 'flex_pinky']

print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

# Finger Thresholds from Data

FINGER_THRESHOLDS = df[flex_cols].median().tolist()
print(f"\n Finger Thresholds (from data): {FINGER_THRESHOLDS}")

# Smoothing & Resampling

def smooth_sequence(sequence, window=3):
    smoothed = np.copy(sequence)
    for i in range(sequence.shape[1]):
        smoothed[:, i] = np.convolve(
            sequence[:, i], np.ones(window)/window, mode='same'
        )
    return smoothed


def resample_sequence(sequence, target_length):
    old_idx = np.linspace(0, 1, len(sequence))
    new_idx = np.linspace(0, 1, target_length)
    out = np.zeros((target_length, sequence.shape[1]))
    for i in range(sequence.shape[1]):
        out[:, i] = np.interp(new_idx, old_idx, sequence[:, i])
    return out


def create_sequences(data_df, seq_length):
    sequences, labels = [], []
    for gesture_id in data_df['Gesture_ID'].unique():
        g        = data_df[data_df['Gesture_ID'] == gesture_id]
        
        #  : تخطي أي حركة سطورها أقل من 5
        if len(g) < 5:
            continue
        
        features = g[feature_cols].values.astype(np.float32)
        label    = g['Label'].iloc[0]
        features = smooth_sequence(features)
        features = resample_sequence(features, seq_length)
        sequences.append(features)
        labels.append(label)
    return np.array(sequences), np.array(labels)



print("\n" + "="*60)
print("Creating stratified gesture-independent 3-way split...")
print("="*60)

df['Unique_Gesture_ID'] = df['Label'].astype(str) + "_" + df['Gesture_ID'].astype(str)

# list of unique_gesture_ids
label_to_ids = defaultdict(list)
for unique_gid in df['Unique_Gesture_ID'].unique():
    lbl = df[df['Unique_Gesture_ID'] == unique_gid]['Label'].iloc[0]
    label_to_ids[lbl].append(unique_gid)

train_ids, val_ids, test_ids = [], [], []
np.random.seed(30) # 
for lbl, ids in label_to_ids.items():
    ids = np.array(ids)
    np.random.shuffle(ids)
    n        = len(ids)
    n_train  = max(1, int(n * 0.70))
    n_val    = max(1, int(n * 0.15))
    
    train_ids.extend(ids[:n_train])
    val_ids.extend(  ids[n_train : n_train + n_val])
    test_ids.extend( ids[n_train + n_val :])

train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)

# split data by g-id
df_train = df[df['Unique_Gesture_ID'].isin(train_ids)].copy()
df_val   = df[df['Unique_Gesture_ID'].isin(val_ids)].copy()
df_test  = df[df['Unique_Gesture_ID'].isin(test_ids)].copy()

print(f"   Train unique gestures : {len(train_ids)}")
print(f"   Val   unique gestures : {len(val_ids)}")
print(f"   Test  unique gestures : {len(test_ids)}")


df_train['Gesture_ID'] = df_train['Unique_Gesture_ID']
df_val['Gesture_ID']   = df_val['Unique_Gesture_ID']
df_test['Gesture_ID']  = df_test['Unique_Gesture_ID']


# Sequences


print("\nBuilding sequences...")
X_train_raw, y_train_raw = create_sequences(df_train, SEQUENCE_LENGTH)
X_val_raw,   y_val_raw   = create_sequences(df_val,   SEQUENCE_LENGTH)
X_test_raw,  y_test_raw  = create_sequences(df_test,  SEQUENCE_LENGTH)

print(f"   X_train : {X_train_raw.shape}")
print(f"   X_val   : {X_val_raw.shape}")
print(f"   X_test  : {X_test_raw.shape}")

df_train.drop(columns=['Unique_Gesture_ID'], inplace=True, errors='ignore')
df_val.drop(columns=['Unique_Gesture_ID'], inplace=True, errors='ignore')
df_test.drop(columns=['Unique_Gesture_ID'], inplace=True, errors='ignore')


# Encode Labels

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_val   = le.transform(y_val_raw)
y_test  = le.transform(y_test_raw)

classes     = le.classes_
num_classes = len(classes)
np.save(CLASSES_PATH, classes)

print(f"\nClasses ({num_classes}): {list(classes)}")

print("\nPer-class split counts:")
for cls in classes:
    ci = list(classes).index(cls)
    tr = (y_train == ci).sum()
    va = (y_val   == ci).sum()
    te = (y_test  == ci).sum()
    print(f"   {cls:15s}   train={tr}   val={va}   test={te}")


# Global Normalisation  (fit on TRAIN only)

n_tr, T, F = X_train_raw.shape

scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_raw.reshape(-1, F))
X_val_flat   = scaler.transform(X_val_raw.reshape(-1, F))
X_test_flat  = scaler.transform(X_test_raw.reshape(-1, F))

X_train = X_train_flat.reshape(n_tr,                   T, F).astype(np.float32)
X_val   = X_val_flat.reshape(  X_val_raw.shape[0],     T, F).astype(np.float32)
X_test  = X_test_flat.reshape( X_test_raw.shape[0],    T, F).astype(np.float32)

joblib.dump(scaler, SCALER_PATH)
print(f"\n Scaler saved → {SCALER_PATH}")
print(f"   mean  (first 3): {scaler.mean_[:3]}")
print(f"   scale (first 3): {scaler.scale_[:3]}")



class_counts  = np.bincount(y_train, minlength=num_classes).astype(np.float32)
class_weights = (1.0 / class_counts)
class_weights = class_weights / class_weights.sum() * num_classes   # normalise
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"\n Class weights: {dict(zip(classes, class_weights.round(3)))}")


# Dataset & DataLoaders

class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):         return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


train_loader = DataLoader(GestureDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(GestureDataset(X_val,   y_val),
                          batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(GestureDataset(X_test,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False)





class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # one linear layer: hidden_size _ 1 score per timestep
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out : (batch, T, hidden_size)
        scores  = self.attention(lstm_out)          # (batch, T, 1)
        weights = torch.softmax(scores, dim=1)      # (batch, T, 1)  sums to 1 over T
        context = (weights * lstm_out).sum(dim=1)   # (batch, hidden_size)
        return context


class SignLanguageLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        #  FIX #4 — attention over all timesteps
        # hidden_size*2 because bidirectional doubles the output size
        self.attention = AttentionPooling(hidden_size * 2)

        self.fc1     = nn.Linear(hidden_size * 2, 64)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)          # (batch, T, hidden*2)
        context     = self.attention(lstm_out)  # (batch, hidden*2)
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


model = SignLanguageLSTM(
    input_size=len(feature_cols),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n Model created  |  Parameters: {total_params:,}")

# Loss (with class weights), Optimiser, Scheduler


#  FIX #6 — weighted loss
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# FIX #7  scheduler now watches VAL loss, not test loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# Training Loop
# FIX #7: early stopping & best-model saving use VAL loss.
#         Test set is NEVER touched during training.


best_val_loss    = float('inf')
patience         = 20
patience_counter = 0

train_losses, val_losses, val_accuracies = [], [], []

print("\n" + "="*60)
print("Training started  (train / val / test — 3-way split)")
print("="*60)

for epoch in range(NUM_EPOCHS):

    # Train 
    model.train()
    total_train_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    #  (val set only) 
    model.eval()
    total_val_loss = 0
    correct = total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:          #  NOT test_loader
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            total_val_loss += criterion(outputs, batch_y).item()
            _, predicted = torch.max(outputs, 1)
            total   += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc      = 100 * correct / total

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    scheduler.step(avg_val_loss)   #  drives LR reduction

    # Save Best 
    if avg_val_loss < best_val_loss:
        best_val_loss    = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

    if patience_counter >= patience:
        print("\nEarly stopping triggered.")
        break


#  (touched for the first time)


model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for batch_X, batch_y in test_loader:          
        outputs  = model(batch_X.to(device))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.numpy())

final_accuracy = accuracy_score(all_labels, all_preds)

print("\n" + "="*60)
print("FINAL TEST RESULTS  (unseen data)")
print("="*60)
print(f"\nTest Accuracy: {final_accuracy*100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds,
                             target_names=classes, digits=3))


# Confusion Matrix ( لخط العربي )


from arabic_reshaper import reshape
from bidi.algorithm import get_display

# تعديل أسماء الكلمات تظهر باللغة العربية  في الرسمة
reshaped_classes = [get_display(reshape(str(cls))) for cls in classes]

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=reshaped_classes, yticklabels=reshaped_classes)

plt.title(get_display(reshape(" Confusion Matrix —  model test")), fontsize=14)
plt.xlabel(get_display(reshape("المتوقع (Predicted)")), fontsize=12)
plt.ylabel(get_display(reshape("الحقيقي (True)")), fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
print("\nconfusion_matrix.png saved with perfect Arabic text!")


#  TTraining Curves

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_losses, label="Train Loss")
axes[0].plot(val_losses,   label="Val Loss")
axes[0].legend(); axes[0].grid(True); axes[0].set_title("Loss")

axes[1].plot(val_accuracies, label="Val Accuracy")
axes[1].legend(); axes[1].grid(True); axes[1].set_title("Val Accuracy")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=300)
print("training_curves.png saved")


# Save Training 


training_info = {
    'sequence_length'  : SEQUENCE_LENGTH,
    'feature_cols'     : feature_cols,
    'classes'          : list(classes),
    'input_size'       : len(feature_cols),
    'hidden_size'      : HIDDEN_SIZE,
    'num_layers'       : NUM_LAYERS,
    'finger_thresholds': FINGER_THRESHOLDS
}

with open(INFO_PATH, 'w', encoding='utf-8') as f:
    json.dump(training_info, f, indent=2, ensure_ascii=False)

print(f"\n✅ training_info.json saved")





print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"  - {MODEL_PATH}")
print(f"  - {CLASSES_PATH}")
print(f"  - {INFO_PATH}")
print(f"  - {SCALER_PATH}")
print(f"  - confusion_matrix.png")
print(f"  - training_curves.png")