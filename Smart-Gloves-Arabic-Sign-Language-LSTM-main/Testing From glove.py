import serial
import time
import numpy as np
import torch
import torch.nn as nn
import serial.tools.list_ports
import json
import pandas as pd
import joblib



DATA_PATH = r"C:\Users\shahd\arabic_dataset"

MODEL_PATH   = f"{DATA_PATH}\\lstm_sign_language_model.pth"
CLASSES_PATH = f"{DATA_PATH}\\classes.npy"
INFO_PATH    = f"{DATA_PATH}\\training_info.json"
DATASET_FILE = f"{DATA_PATH}\\sign_language_dataset.csv"
SCALER_PATH  = f"{DATA_PATH}\\scaler.pkl"

BAUD_RATE            = 115200
CONFIDENCE_THRESHOLD = 0.35
REQUIRED_DURATION    = 1.5
MOTION_THRESHOLD     = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



print("\n🔄 Loading model files...")

with open(INFO_PATH, 'r', encoding='utf-8') as f:
    training_info = json.load(f)

SEQUENCE_LENGTH = training_info['sequence_length']   # reads 32 automatically
feature_cols    = training_info['feature_cols']
input_size      = len(feature_cols)
classes         = np.load(CLASSES_PATH, allow_pickle=True)

#   Load thresholds computed from training data
# (no more hardcoded [100, 100, 100, 100, 100])
FINGER_THRESHOLDS = training_info['finger_thresholds']

print(f" Finger Thresholds loaded from training: {FINGER_THRESHOLDS}")

# Load the scaler fitted during training.
# This is the ONLY normalisation applied to live data.

scaler = joblib.load(SCALER_PATH)

print(" Scaler loaded")
print(f"   mean  (first 3): {scaler.mean_[:3]}")
print(f"   scale (first 3): {scaler.scale_[:3]}")
print(f" Classes         : {list(classes)}")
print(f" Sequence Length : {SEQUENCE_LENGTH}")


#  Hybrid Rules


df_dataset = pd.read_csv(DATASET_FILE)
print(" Dataset loaded for hybrid system")

# Finger Pattern 

def get_finger_pattern(row):
    flex_values = [
        row['flex_thumb'], row['flex_index'], row['flex_middle'],
        row['flex_ring'],  row['flex_pinky']
    ]
    return tuple(flex_values[i] < FINGER_THRESHOLDS[i] for i in range(5))

# Build Word Signatures


df_dataset['finger_pattern'] = df_dataset.apply(get_finger_pattern, axis=1)

word_signatures = {}
for word in classes:
    word_data = df_dataset[df_dataset['Label'] == word]
    most_common = word_data['finger_pattern'].value_counts().index[0]
    word_signatures[word] = {'finger_pattern': most_common}

print("Finger signatures created")


class AttentionPooling(nn.Module):
    """Learns which timesteps matter instead of averaging all equally."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out : (batch, T, hidden_size)
        scores  = self.attention(lstm_out)         # (batch, T, 1)
        weights = torch.softmax(scores, dim=1)     # (batch, T, 1)
        context = (weights * lstm_out).sum(dim=1)  # (batch, hidden_size)
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

        # hidden_size*2 because bidirectional doubles the output
        self.attention = AttentionPooling(hidden_size * 2)

        self.fc1     = nn.Linear(hidden_size * 2, 64)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context     = self.attention(lstm_out)   # attention over all timesteps
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Load Model


model = SignLanguageLSTM(
    input_size=input_size,
    hidden_size=training_info['hidden_size'],
    num_layers=training_info['num_layers'],
    num_classes=len(classes),
    dropout=0.2
).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()
print(" Model loaded successfully")

# Connect Glove


def connect_glove():
    ports = serial.tools.list_ports.comports()

    print("\n Available Ports:")
    for i, port in enumerate(ports):
        print(f"{i+1}. {port.device} - {port.description}")

    if not ports:
        return None

    choice = input("\n Select port: ").strip()
    port_name = ports[int(choice)-1].device if choice.isdigit() else choice

    try:
        ser = serial.Serial(port_name, BAUD_RATE, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f" Connected to {port_name}")
        return ser
    except Exception as e:
        print(f" Connection failed: {e}")
        return None


# Parse Serial Data

def parse_glove_data(line):
    try:
        line = line.strip()
        if line.startswith("DATA,"):
            line = line[5:]
        values = [float(x) for x in line.split(',')]
        if   len(values) == 14: return values[1:]
        elif len(values) == 13: return values
        else:                   return None
    except:
        return None
# Connect Glove 


WINDOW_SIZE         = 5   # look-back window (frames)
SUSTAINED_THRESHOLD = 3   # minimum active frames inside the window

def wait_for_gesture_start(ser):
    print("\n Waiting for gesture...")

    motion_buffer = []

    while True:
        if ser.in_waiting > 0:
            line   = ser.readline().decode('utf-8', errors='ignore').strip()
            values = parse_glove_data(line)

            if values and len(values) == input_size:
                ax = abs(values[5])
                ay = abs(values[6])
                az = abs(values[7])
                motion = ax + ay + az

                motion_buffer.append(motion > MOTION_THRESHOLD)

                # keep only the last WINDOW_SIZE readings
                if len(motion_buffer) > WINDOW_SIZE:
                    motion_buffer.pop(0)

                if sum(motion_buffer) >= SUSTAINED_THRESHOLD:
                    print("🚀 Gesture detected!")
                    return

        time.sleep(0.01)

# Collect Samples

def collect_gesture_data(ser):
    samples    = []
    start_time = time.time()

    print(f" Collecting {REQUIRED_DURATION} sec...")

    while time.time() - start_time < REQUIRED_DURATION:
        if ser.in_waiting > 0:
            line   = ser.readline().decode('utf-8', errors='ignore').strip()
            values = parse_glove_data(line)
            if values and len(values) == input_size:
                samples.append(values)
        time.sleep(0.005)

    print(f"Samples collected: {len(samples)}")
    return np.array(samples, dtype=np.float32) if samples else None

# Smooth & Resample

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

# Extract Live Finger Pattern

def extract_live_finger_pattern(samples):
    avg_flex = np.mean(samples[:, :5], axis=0)
    print(f"\n Average Flex Values: {avg_flex}")
    return tuple(avg_flex[i] < FINGER_THRESHOLDS[i] for i in range(5))


def hamming_distance(p1, p2):
    return sum(a != b for a, b in zip(p1, p2))


def get_candidate_words(samples):
    live_pattern = extract_live_finger_pattern(samples)
    print(f"\n Live Finger Pattern: {live_pattern}")

    # exact match 
    candidates = [
        word for word, sig in word_signatures.items()
        if sig['finger_pattern'] == live_pattern
    ]

    if candidates:
        print(f" Exact match — Candidates: {candidates}")
        return candidates

    # Hamming-distance fallback 
    print("No exact match — finding closest pattern by Hamming distance...")

    scored = [
        (word, hamming_distance(live_pattern, sig['finger_pattern']))
        for word, sig in word_signatures.items()
    ]
    scored.sort(key=lambda x: x[1])

    min_dist   = scored[0][1]
    # accept words within distance of min+1
    candidates = [w for w, d in scored if d <= min_dist + 1]

    print(f" Closest Hamming distance: {min_dist}")
    print(f" Candidates after fallback: {candidates}")
    return candidates

# Preprocessing


def preprocess_sequence(samples):
    samples = smooth_sequence(samples)
    samples = resample_sequence(samples, SEQUENCE_LENGTH)
    samples = scaler.transform(samples)
    return samples.astype(np.float32)

# Prediction

def predict(samples):

    #  filter BEFORE normalisation
    candidates = get_candidate_words(samples)



    # preprocess
    samples = preprocess_sequence(samples)

    input_tensor = torch.tensor(samples, dtype=torch.float32) \
                        .unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs  = torch.softmax(output, dim=1)

    all_probs = probs[0].cpu().numpy()

  
    raw_top5 = sorted(
        enumerate(all_probs), key=lambda x: x[1], reverse=True
    )[:5]
    print("\n Raw LSTM top-5 (before hybrid filter):")
    for idx, prob in raw_top5:
        print(f"  {classes[idx]:15s}: {prob*100:.2f}%")

    # filter to candidates only
    filtered = [
        (word, all_probs[list(classes).index(word)])
        for word in candidates
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)

    predicted_word = filtered[0][0]
    confidence     = filtered[0][1]

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_word = "Unknown"

    print("\nCandidate Probabilities:\n")
    for word, prob in filtered[:5]:
        print(f"  {word:15s}: {prob*100:.2f}%")

    return predicted_word, confidence

# Main

def main():
    print("\n" + "="*60)
    print("Hybrid Real-Time Sign Language Recognition")
    print("="*60)

    ser = connect_glove()
    if ser is None:
        return

    print("\n System started")
    print(" Press Ctrl+C to stop")

    prediction_count = 0

    try:
        while True:

            wait_for_gesture_start(ser)
            ser.reset_input_buffer()

            samples = collect_gesture_data(ser)
            if samples is None:
                continue

            prediction_count += 1
            print("\n" + "="*60)
            print(f" Prediction #{prediction_count}")
            print("="*60)

            predicted_word, confidence = predict(samples)

            print("\n" + "="*60)
            print(f" Predicted Word : {predicted_word}")
            print(f" Confidence     : {confidence*100:.2f}%")
            print("="*60)

            print("\n Waiting 2 sec...\n")
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n System stopped")

    finally:
        ser.close()
        print(f"\n Total Predictions: {prediction_count}")

# Run
if __name__ == "__main__":
    main()