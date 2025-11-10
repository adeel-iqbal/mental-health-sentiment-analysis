import pickle
from keras.models import load_model

# Paths to artifacts
MODEL_PATH = "artifacts/best_model.keras"
TOKENIZER_PATH = "artifacts/tokenizer.pkl"
LABEL_ENCODER_PATH = "artifacts/label_encoder.pkl"
MAX_LEN_PATH = "artifacts/max_len.pkl"

# Load trained model
model = load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Load max_len
with open(MAX_LEN_PATH, "rb") as f:
    max_len = pickle.load(f)

print("Model and artifacts loaded successfully!")
