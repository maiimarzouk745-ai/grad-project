import numpy as np
import torch
import torch.nn as nn
import joblib

MODEL_PATH = "lstm_sign_language_model.pth"
SCALER_PATH = "scaler.pkl"
CLASSES_PATH = "classes.npy"

SEQUENCE_LENGTH = 64   # Numbre of frames
INPUT_SIZE = 13        # Numbre of features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(SignLanguageLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.attention = nn.Linear(128, 1)
        self.fc1 = nn.Linear(128, 32)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        self.bn = nn.BatchNorm1d(32)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        out = self.fc1(context)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

#  Load Model 
scaler = joblib.load(SCALER_PATH)
classes = np.load(CLASSES_PATH)

model = SignLanguageLSTM(INPUT_SIZE, 64, 2, len(classes)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded successfully!")
print("Classes:", classes)


print("\nEnter your sequence data")
print(f"You need {SEQUENCE_LENGTH} rows, each with {INPUT_SIZE} values")
print("Format: value1,value2,...,value13\n")

sequence = []

for i in range(SEQUENCE_LENGTH):
    row = input(f"Row {i+1}: ")
    values = list(map(float, row.strip().split(",")))
    
    if len(values) != INPUT_SIZE:
        print("Invalid input length!")
        exit()
    
    sequence.append(values)

sequence = np.array(sequence)

# reshape for scaling
seq_scaled = scaler.transform(sequence)
seq_scaled = seq_scaled.reshape(1, SEQUENCE_LENGTH, INPUT_SIZE)

#  Prediction
with torch.no_grad():
    x = torch.tensor(seq_scaled, dtype=torch.float32).to(device)
    outputs = model(x)
    
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)

predicted_class = classes[predicted.item()]
confidence = confidence.item()

#  Out 
print("\nPrediction Result:")
print(f"Predicted Word: {predicted_class}")
print(f"Confidence: {confidence:.4f}")

# Optional threshold
if confidence < 0.7:
    print("Low confidence >>> Possibly no gesture")