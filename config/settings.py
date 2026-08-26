"""
Global configuration for the Digital Content Virality Predictor.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────────────────
DATASET_SIZE = 1_000_000          # Number of synthetic rows
DATASET_FILENAME = "virality_dataset.parquet"
DAILY_DATA_FILENAME = "daily_feed_{date}.parquet"

# ─── Platforms & Regions ─────────────────────────────────────────────
PLATFORMS = ["Twitter", "YouTube", "Instagram", "TikTok", "Facebook", "LinkedIn", "Reddit"]

COUNTRIES = [
    "US", "UK", "IN", "BR", "DE", "FR", "JP", "KR", "MX", "CA",
    "AU", "NG", "EG", "ZA", "SA", "AE", "PK", "BD", "ID", "PH",
    "TR", "IT", "ES", "NL", "SE", "PL", "TH", "VN", "CO", "AR",
]

LANGUAGES = [
    "en", "es", "pt", "hi", "ar", "fr", "de", "ja", "ko", "zh",
    "it", "nl", "sv", "pl", "tr", "th", "vi", "id", "tl", "bn",
    "ur", "ru", "fa", "sw", "ha",
]

CONTENT_TYPES = ["image", "video", "text", "carousel", "story", "reel", "live"]

CATEGORIES = [
    "Entertainment", "Education", "Technology", "Sports", "Music",
    "Gaming", "Fashion", "Food", "Travel", "Health", "Finance",
    "News", "Comedy", "Science", "Art", "Politics", "Lifestyle",
    "Motivation", "Beauty", "Pets",
]

# ─── Virality classes ────────────────────────────────────────────────
VIRALITY_THRESHOLDS = {
    "Low":    (0, 10_000),
    "Medium": (10_000, 500_000),
    "High":   (500_000, float("inf")),
}
VIRALITY_CLASSES = list(VIRALITY_THRESHOLDS.keys())
NUM_VIRALITY_CLASSES = len(VIRALITY_CLASSES)

# ─── Model ───────────────────────────────────────────────────────────
TEXT_EMBEDDING_DIM = 384          # sentence-transformers all-MiniLM-L6-v2
IMAGE_EMBEDDING_DIM = 512         # ResNet-18 penultimate layer
TABULAR_FEATURE_DIM = 0          # set dynamically after preprocessing
HIDDEN_DIM = 256
DROPOUT = 0.3

# ─── Training ────────────────────────────────────────────────────────
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
WEIGHT_DECAY = 1e-4

# ─── Incremental learning ───────────────────────────────────────────
INCREMENTAL_LR = 5e-5
INCREMENTAL_EPOCHS = 3
EWC_LAMBDA = 0.4                 # Elastic Weight Consolidation strength

# ─── Text model ──────────────────────────────────────────────────────
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# ─── Image model ─────────────────────────────────────────────────────
IMAGE_CNN_MODEL = "resnet18"
IMAGE_SIZE = (224, 224)

# ─── API Keys (set via environment) ──────────────────────────────────
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
INSTAGRAM_USERNAME = os.getenv("INSTAGRAM_USERNAME", "")
INSTAGRAM_PASSWORD = os.getenv("INSTAGRAM_PASSWORD", "")

# ─── Streamlit ───────────────────────────────────────────────────────
STREAMLIT_PAGE_TITLE = "🚀 Digital Content Virality Predictor"
STREAMLIT_PAGE_ICON = "🔮"
STREAMLIT_LAYOUT = "wide"

# ─── Logging ─────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
