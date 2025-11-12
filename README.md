# Mental Health Sentiment Analysis using Bidirectional GRU

A Deep Learning-based microservice for classifying mental health sentiment from text data. This project uses a Bidirectional GRU (Gated Recurrent Unit) neural network to analyze text statements and predict mental health categories with high accuracy.

## 🎯 Overview

This project provides a FastAPI-based REST API microservice that predicts mental health sentiment categories from user-provided text. The model achieves **84.83% validation accuracy** during training and **94% test accuracy** on evaluation, and can classify text into seven distinct mental health categories.

### Supported Categories
- **Normal** - General, non-clinical statements
- **Depression** - Indicators of depressive symptoms
- **Anxiety** - Signs of anxiety or worry
- **Suicidal** - Thoughts related to self-harm (requires immediate professional attention)
- **Bipolar** - Mood fluctuation patterns
- **Stress** - Stress-related concerns
- **Personality Disorder** - Personality-related issues

## 🚀 Features

- **High Accuracy**: 94% test accuracy with 84.83% validation accuracy
- **Fast Inference**: Real-time prediction via REST API
- **Bidirectional GRU Architecture**: Captures context from both directions in text
- **Data Augmentation**: Enhanced training with synonym replacement and random insertion (EDA techniques)
- **Production Ready**: Microservice architecture with FastAPI
- **Easy Integration**: Simple JSON-based API for seamless integration

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 94.17% |
| Validation Accuracy | 84.83% |
| Test Accuracy | 94.00% |
| Weighted F1-Score (Test) | 0.94 |

### Per-Class Performance (Test Set Evaluation)

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Anxiety | 0.98 | 0.99 | 0.99 |
| Bipolar | 0.98 | 0.99 | 0.99 |
| Depression | 0.92 | 0.84 | 0.88 |
| Normal | 0.98 | 0.98 | 0.98 |
| Personality Disorder | 0.94 | 0.98 | 0.96 |
| Stress | 0.93 | 0.99 | 0.96 |
| Suicidal | 0.84 | 0.89 | 0.86 |

## 🏗️ Architecture

### Model Architecture
- **Embedding Layer**: 100-dimensional word embeddings (vocabulary size: 79,813)
- **Bidirectional GRU**: 64 units processing sequences in both directions
- **Dropout Layer**: 0.3 dropout rate for regularization
- **Dense Output Layer**: 7-class softmax activation

### Project Structure
```
mental-health-sentiment-analysis/
├── app/
│   ├── main.py              # FastAPI application and endpoints
│   ├── model_loader.py      # Model and artifact loading
│   └── predictor.py         # Prediction logic
├── artifacts/
│   ├── best_model.keras     # Trained Bidirectional GRU model
│   ├── tokenizer.pkl        # Text tokenizer
│   ├── label_encoder.pkl    # Label encoder for classes
│   └── max_len.pkl          # Maximum sequence length (6300)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/adeel-iqbal/mental-health-sentiment-analysis.git
cd mental-health-sentiment-analysis
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify artifacts**
Ensure all model artifacts are present in the `artifacts/` directory:
- `best_model.keras`
- `tokenizer.pkl`
- `label_encoder.pkl`
- `max_len.pkl`

## 🚦 Usage

### Starting the API Server

Run the FastAPI server using uvicorn:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```bash
GET /
```

**Response:**
```json
{
  "message": "ML Service is running..."
}
```

#### 2. Predict Mental Health Sentiment
```bash
POST /predict
```

**Request Body:**
```json
{
  "text": "I can't sleep at night and feel anxious all day."
}
```

**Response:**
```json
{
  "predicted_class": "Anxiety",
  "confidence": 0.9967
}
```

### Example Usage with cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I feel like everything is meaningless."}'
```

### Example Usage with Python

```python
import requests

url = "http://localhost:8000/predict"
payload = {"text": "Just chilling and having a peaceful weekend."}

response = requests.post(url, json=payload)
print(response.json())
# Output: {"predicted_class": "Normal", "confidence": 0.6048}
```

### Interactive API Documentation

FastAPI provides automatic interactive documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📦 Dependencies

```
fastapi==0.109.0
uvicorn==0.27.0
tensorflow==2.15.0
keras==2.15.0
numpy==1.24.3
pydantic==2.5.0
scikit-learn==1.3.2
```

See `requirements.txt` for the complete list of dependencies.

## 🔬 Training Details

### Dataset
- **Source**: Kaggle - Sentiment Analysis for Mental Health
- **Original Size**: 53,043 samples
- **After Cleaning**: 51,093 samples (removed 362 null values and 1,588 duplicates)
- **After Augmentation**: 69,723 samples

### Data Preprocessing
1. **Text Cleaning**: Lowercase conversion, special character removal, whitespace normalization
2. **Data Augmentation**: Applied EDA (Easy Data Augmentation) techniques to minority classes
   - Synonym Replacement (SR)
   - Random Insertion (RI)
3. **Tokenization**: Vocabulary size of 79,813 unique words
4. **Padding**: Sequences padded to max length of 6,300 tokens

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 64
- **Epochs**: 5 (with early stopping)
- **Validation Split**: 10%
- **Callbacks**: ModelCheckpoint, EarlyStopping (patience=3)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Adeel Iqbal Memon**
- Email: [adeelmemon096@yahoo.com](mailto:adeelmemon096@yahoo.com)
- LinkedIn: [linkedin.com/in/adeeliqbalmemon](https://linkedin.com/in/adeeliqbalmemon)
- GitHub: [@adeel-iqbal](https://github.com/adeel-iqbal)

## 🙏 Acknowledgments

- Dataset: [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) on Kaggle
- FastAPI framework for the microservice architecture
- TensorFlow/Keras for deep learning capabilities
- The mental health community for raising awareness

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time streaming prediction
- [ ] Docker containerization
- [ ] Deployment to cloud platforms (AWS, GCP, Azure)
- [ ] Integration with mental health chatbots
- [ ] Fine-tuning with transformer models (BERT, RoBERTa)

---

⭐ **If you find this project helpful, please consider giving it a star on GitHub!**
