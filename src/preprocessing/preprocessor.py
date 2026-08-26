"""
Preprocessing pipeline for all modalities:
  - Text  → sentence-transformer embeddings
  - Image → ResNet-18 CNN embeddings (simulated for synthetic data)
  - Tabular → label encoding + standard scaling
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DATA_DIR, TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM,
    SENTENCE_TRANSFORMER_MODEL, VIRALITY_CLASSES,
)
from src.utils.helpers import get_logger

logger = get_logger("preprocessor")


# ═══════════════════════════════════════════════════════════════════════
#  TEXT EMBEDDING
# ═══════════════════════════════════════════════════════════════════════
class TextEmbedder:
    """Generate sentence embeddings using a SentenceTransformer model."""

    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Using simulated text embeddings."
                )
                self._model = "simulated"

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Return (N, dim) embedding matrix."""
        self._load_model()
        if self._model == "simulated":
            return self._simulated_embeddings(texts)
        embeddings = self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def _simulated_embeddings(texts: list[str]) -> np.ndarray:
        """Fallback: generate deterministic pseudo-embeddings from text length."""
        rng = np.random.default_rng(42)
        embs = np.zeros((len(texts), TEXT_EMBEDDING_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            seed_val = hash(t) % (2**31)
            local_rng = np.random.default_rng(seed_val)
            embs[i] = local_rng.normal(0, 0.3, TEXT_EMBEDDING_DIM)
        return embs


# ═══════════════════════════════════════════════════════════════════════
#  IMAGE EMBEDDING
# ═══════════════════════════════════════════════════════════════════════
class ImageEmbedder:
    """Generate image embeddings using ResNet-18 (or simulated)."""

    def __init__(self):
        self._model = None
        self._transform = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torchvision.models as models
            import torchvision.transforms as transforms
            logger.info("Loading ResNet-18 for image embeddings")
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Remove final FC layer — keep feature extractor
            self._model = torch.nn.Sequential(*list(resnet.children())[:-1])
            self._model.eval()
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        except Exception as e:
            logger.warning(f"Cannot load ResNet-18: {e}. Using simulated image embeddings.")
            self._model = "simulated"

    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        """Embed a list of image file paths."""
        self._load_model()
        if self._model == "simulated":
            return self._simulated(len(image_paths))
        from PIL import Image
        embeddings = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor = self._transform(img).unsqueeze(0)
                with torch.no_grad():
                    feat = self._model(tensor).squeeze().numpy()
                embeddings.append(feat)
            except Exception:
                embeddings.append(np.zeros(IMAGE_EMBEDDING_DIM, dtype=np.float32))
        return np.array(embeddings, dtype=np.float32)

    @staticmethod
    def _simulated(n: int) -> np.ndarray:
        """Generate simulated image embeddings."""
        rng = np.random.default_rng(123)
        return rng.normal(0, 0.25, (n, IMAGE_EMBEDDING_DIM)).astype(np.float32)

    def embed_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Generate embeddings based on the img_emb_mean column (synthetic data)."""
        n = len(df)
        rng = np.random.default_rng(123)
        base = rng.normal(0, 0.25, (n, IMAGE_EMBEDDING_DIM)).astype(np.float32)
        if "img_emb_mean" in df.columns:
            # Bias the first dimension with the mean to keep signal
            base[:, 0] = df["img_emb_mean"].values.astype(np.float32)
        return base


# ═══════════════════════════════════════════════════════════════════════
#  TABULAR PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════
class TabularPreprocessor:
    """Label-encode categoricals and standard-scale numerics."""

    CATEGORICAL_COLS = [
        "platform", "country", "language", "content_type", "category",
    ]

    NUMERICAL_COLS = [
        "follower_count", "hist_engagement_rate", "posting_hour",
        "posting_day", "account_age_days", "caption_length", "n_hashtags",
        "is_verified", "has_image", "has_video", "has_cta", "has_url",
        "is_reply", "mentions_count", "emoji_count", "sentiment",
        "prev_avg_views", "text_emb_mean", "img_emb_mean",
    ]

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit encoders and scaler on training data."""
        for col in self.CATEGORICAL_COLS:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le

        num_data = df[self.NUMERICAL_COLS].fillna(0).values.astype(np.float32)
        self.scaler.fit(num_data)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform and return concatenated tabular features."""
        if not self.fitted:
            raise RuntimeError("Call fit() before transform().")

        # Encode categoricals
        cat_encoded = []
        for col in self.CATEGORICAL_COLS:
            le = self.label_encoders[col]
            vals = df[col].astype(str).copy()
            # Handle unseen labels
            mask = ~vals.isin(le.classes_)
            vals[mask] = le.classes_[0]
            cat_encoded.append(le.transform(vals).reshape(-1, 1))
        cat_matrix = np.hstack(cat_encoded).astype(np.float32)

        # Scale numericals
        num_data = df[self.NUMERICAL_COLS].fillna(0).values.astype(np.float32)
        num_scaled = self.scaler.transform(num_data).astype(np.float32)

        return np.hstack([cat_matrix, num_scaled])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def feature_dim(self) -> int:
        return len(self.CATEGORICAL_COLS) + len(self.NUMERICAL_COLS)

    @property
    def feature_names(self) -> list[str]:
        return self.CATEGORICAL_COLS + self.NUMERICAL_COLS

    def save(self, path: Path) -> None:
        joblib.dump({
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
        }, path)
        logger.info(f"Tabular preprocessor saved to {path}")

    def load(self, path: Path) -> "TabularPreprocessor":
        data = joblib.load(path)
        self.label_encoders = data["label_encoders"]
        self.scaler = data["scaler"]
        self.fitted = True
        logger.info(f"Tabular preprocessor loaded from {path}")
        return self


# ═══════════════════════════════════════════════════════════════════════
#  TARGET ENCODER
# ═══════════════════════════════════════════════════════════════════════
class TargetEncoder:
    """Encode virality class labels and scale regression target."""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.view_scaler = StandardScaler()
        self.fitted = False

    def fit(self, views: np.ndarray, virality: np.ndarray) -> "TargetEncoder":
        log_views = np.log1p(views).reshape(-1, 1)
        self.view_scaler.fit(log_views)
        self.label_encoder.fit(VIRALITY_CLASSES)
        self.fitted = True
        return self

    def transform_regression(self, views: np.ndarray) -> np.ndarray:
        log_views = np.log1p(views).reshape(-1, 1)
        return self.view_scaler.transform(log_views).flatten().astype(np.float32)

    def inverse_transform_regression(self, scaled: np.ndarray) -> np.ndarray:
        unscaled = self.view_scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()
        return np.expm1(unscaled)

    def transform_classification(self, virality: np.ndarray) -> np.ndarray:
        return self.label_encoder.transform(virality).astype(np.int64)

    def inverse_transform_classification(self, encoded: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(encoded)

    def save(self, path: Path) -> None:
        joblib.dump({
            "label_encoder": self.label_encoder,
            "view_scaler": self.view_scaler,
        }, path)

    def load(self, path: Path) -> "TargetEncoder":
        data = joblib.load(path)
        self.label_encoder = data["label_encoder"]
        self.view_scaler = data["view_scaler"]
        self.fitted = True
        return self


# ═══════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════
def preprocess_dataset(
    df: pd.DataFrame,
    text_embedder: TextEmbedder | None = None,
    image_embedder: ImageEmbedder | None = None,
    tabular_preprocessor: TabularPreprocessor | None = None,
    target_encoder: TargetEncoder | None = None,
    fit: bool = True,
    use_simulated_embeddings: bool = True,
) -> dict:
    """
    Full preprocessing: returns dict of numpy arrays ready for PyTorch.
    """
    logger.info(f"Preprocessing {len(df):,} rows …")

    # ── Text embeddings ──────────────────────────────────────────
    if use_simulated_embeddings:
        logger.info("Using simulated text embeddings (fast mode)")
        rng = np.random.default_rng(42)
        text_emb = rng.normal(0, 0.3, (len(df), TEXT_EMBEDDING_DIM)).astype(np.float32)
        if "text_emb_mean" in df.columns:
            text_emb[:, 0] = df["text_emb_mean"].values.astype(np.float32)
    else:
        if text_embedder is None:
            text_embedder = TextEmbedder()
        combined_text = (
            df["caption"].fillna("") + " " +
            df["hashtags"].fillna("") + " " +
            df["description"].fillna("")
        ).tolist()
        text_emb = text_embedder.embed(combined_text)

    # ── Image embeddings ─────────────────────────────────────────
    if use_simulated_embeddings:
        logger.info("Using simulated image embeddings (fast mode)")
        if image_embedder is None:
            image_embedder = ImageEmbedder()
        img_emb = image_embedder.embed_from_dataframe(df)
    else:
        if image_embedder is None:
            image_embedder = ImageEmbedder()
        img_emb = image_embedder.embed_from_dataframe(df)

    # ── Tabular features ─────────────────────────────────────────
    if tabular_preprocessor is None:
        tabular_preprocessor = TabularPreprocessor()
    if fit:
        tab_features = tabular_preprocessor.fit_transform(df)
    else:
        tab_features = tabular_preprocessor.transform(df)

    # ── Targets ──────────────────────────────────────────────────
    if target_encoder is None:
        target_encoder = TargetEncoder()
    views = df["views"].values.astype(np.float64)
    virality = df["virality_class"].values

    if fit:
        target_encoder.fit(views, virality)

    y_reg = target_encoder.transform_regression(views)
    y_cls = target_encoder.transform_classification(virality)

    logger.info(
        f"Preprocessed: text_emb={text_emb.shape}, img_emb={img_emb.shape}, "
        f"tab={tab_features.shape}, y_reg={y_reg.shape}, y_cls={y_cls.shape}"
    )

    return {
        "text_emb": text_emb,
        "img_emb": img_emb,
        "tabular": tab_features,
        "y_regression": y_reg,
        "y_classification": y_cls,
        "tabular_preprocessor": tabular_preprocessor,
        "target_encoder": target_encoder,
        "text_embedder": text_embedder,
        "image_embedder": image_embedder,
    }
