# 🚀 Digital Content Virality Predictor

An AI-powered multi-modal system that predicts social media content reach and classifies virality using deep learning. Built with **PyTorch**, featuring real-time data collection, incremental learning, and an interactive **Streamlit** dashboard.

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Multi-Modal Input** | Text (captions/hashtags) + Image + Video + Tabular features |
| **Dual Prediction** | Regression (views/reach) + Classification (Low/Medium/High virality) |
| **1M+ Dataset** | Synthetic dataset with realistic correlations across 7 platforms |
| **PyTorch Model** | Attention-based fusion network with uncertainty-weighted multi-task loss |
| **Incremental Learning** | Elastic Weight Consolidation (EWC) for daily model updates |
| **Real-Time Collection** | Twitter, YouTube, Instagram, TikTok data collectors |
| **Global Scope** | 30 countries, 25 languages, 20 content categories |
| **Interactive UI** | Streamlit app with 7 pages: Dashboard, Predict, Analytics, Trends, etc. |

---

## 🏗️ Architecture

```
┌─────────┐   ┌─────────┐   ┌──────────┐
│  Text   │   │  Image  │   │ Tabular  │
│ Encoder │   │ Encoder │   │ Encoder  │
│ (384-d) │   │ (512-d) │   │ (24-d)   │
└────┬────┘   └────┬────┘   └────┬─────┘
     │             │             │
     └──────┬──────┘─────────────┘
            │  Attention Fusion
      ┌─────┴─────┐
      │  Fusion   │
      │  Network  │
      └─────┬─────┘
            │
   ┌────────┴────────┐
   │                 │
┌──┴───┐        ┌───┴────┐
│ Reg  │        │  Cls   │
│ Head │        │  Head  │
└──────┘        └────────┘
(views)       (Low/Med/High)
```

---

## 📁 Project Structure

```
Digital-Content-Virality-Predictor/
├── config/
│   ├── __init__.py
│   └── settings.py                # All hyperparameters & configuration
├── src/
│   ├── data_generation/
│   │   └── generate_dataset.py    # 1M+ row synthetic dataset generator
│   ├── data_collection/
│   │   └── collectors.py          # Twitter, YouTube, IG, TikTok collectors
│   ├── preprocessing/
│   │   └── preprocessor.py        # Text/Image/Tabular preprocessing
│   ├── model/
│   │   └── virality_model.py      # Multi-input PyTorch model
│   ├── training/
│   │   └── trainer.py             # Training, evaluation, EWC incremental
│   └── utils/
│       └── helpers.py             # Logging, seeding, device utilities
├── app/
│   └── streamlit_app.py           # Full Streamlit application (7 pages)
├── notebooks/
│   └── virality_predictor_workflow.ipynb  # Complete workflow notebook
├── data/                          # Generated & collected datasets
├── models/                        # Saved PyTorch models
├── logs/                          # Training & collection logs
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
# Full 1M rows
python -m src.data_generation.generate_dataset --rows 1000000

# Quick test with 100K rows
python -m src.data_generation.generate_dataset --rows 100000
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/virality_predictor_workflow.ipynb
```

This notebook walks through the complete pipeline: dataset → preprocessing → training → evaluation.

### 4. Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Model Performance

| Metric | Task | Expected Range |
|--------|------|----------------|
| **R² Score** | Regression | 0.65 – 0.85 |
| **MAE** | Regression | 5K – 50K views |
| **RMSE** | Regression | 15K – 100K views |
| **Accuracy** | Classification | 0.70 – 0.88 |
| **F1 (Weighted)** | Classification | 0.68 – 0.85 |

---

## 🔄 Incremental Learning Workflow

The system uses **Elastic Weight Consolidation (EWC)** for daily model updates:

1. **Collect** daily data from Twitter, YouTube, Instagram, TikTok
2. **Validate** — remove spam, duplicates, anomalies
3. **Compute Fisher** Information Matrix on existing training data
4. **Fine-tune** model on new data with EWC regularization
5. **Evaluate** to ensure no regression on historical data

```python
from src.training.trainer import EWCTrainer

ewc = EWCTrainer(model, device, ewc_lambda=0.4)
ewc.compute_fisher(old_data_loader)
ewc.incremental_train(new_loader, val_loader, num_epochs=3, lr=5e-5)
```

---

## 🌍 Global Scope

- **30 Countries**: US, UK, IN, BR, DE, FR, JP, KR, and more
- **25 Languages**: en, es, pt, hi, ar, fr, de, ja, ko, zh, and more
- **7 Platforms**: Twitter, YouTube, Instagram, TikTok, Facebook, LinkedIn, Reddit
- **20 Categories**: Entertainment, Education, Technology, Sports, etc.

---

## 🔑 API Configuration (Optional)

Set these environment variables for real data collection:

```bash
export TWITTER_BEARER_TOKEN=your_token
export YOUTUBE_API_KEY=your_key
export INSTAGRAM_USERNAME=your_username
export INSTAGRAM_PASSWORD=your_password
```

Without API keys, the system uses realistic simulated data.

---

## ✅ Deployment Checklist

- [x] Dataset generated (1M+ rows)
- [x] Multi-modal preprocessing pipeline
- [x] Multi-input PyTorch model
- [x] Training with early stopping & cosine LR
- [x] Evaluation: R², MAE, RMSE, Accuracy, F1
- [x] EWC incremental learning
- [x] Data collectors (4 platforms)
- [x] Streamlit UI (7 pages)
- [x] Feature importance analysis
- [x] Error handling & validation
- [x] Global scope (30 countries, 25 languages)
- [x] Saved model artifacts
- [x] Jupyter Notebook documentation
- [x] Production-ready code structure

---

## 📜 License

MIT License
"# Digital-Content-Virality-Predictor" 
