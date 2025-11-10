import numpy as np
from keras.preprocessing.sequence import pad_sequences
from app.model_loader import model, tokenizer, label_encoder, max_len

def predict_text(text: str):
    """
    Input: raw text string
    Output: predicted class label + confidence
    """
    # Lowercase
    text = text.lower()

    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    # Predict
    pred_probs = model.predict(padded)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    pred_label = label_encoder.inverse_transform([pred_index])[0]
    confidence = float(np.max(pred_probs))

    return {
        "predicted_class": pred_label,
        "confidence": round(confidence, 4)
    }
