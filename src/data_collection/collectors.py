"""
Real-time social media data collectors (skeleton implementations).

In production, these would connect to actual APIs with valid credentials.
For development, they generate realistic simulated daily reports.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CATEGORIES, CONTENT_TYPES, COUNTRIES, LANGUAGES, PLATFORMS,
    RAW_DATA_DIR, TWITTER_BEARER_TOKEN, VIRALITY_THRESHOLDS,
    YOUTUBE_API_KEY,
)
from src.utils.helpers import get_logger, virality_label

logger = get_logger("data_collector")


# ═══════════════════════════════════════════════════════════════════════
#  BASE COLLECTOR
# ═══════════════════════════════════════════════════════════════════════
class BaseCollector:
    """Abstract base for platform data collectors."""

    platform: str = "Unknown"

    def collect(self, date: str | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out spam/fake data."""
        # Remove rows with suspiciously low engagement
        if "views" in df.columns:
            df = df[df["views"] >= 0]
        # Remove duplicates
        if "post_id" in df.columns:
            df = df.drop_duplicates(subset=["post_id"])
        # Remove rows with empty captions
        if "caption" in df.columns:
            df = df[df["caption"].str.strip().str.len() > 0]
        return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
#  TWITTER COLLECTOR
# ═══════════════════════════════════════════════════════════════════════
class TwitterCollector(BaseCollector):
    platform = "Twitter"

    def collect(self, date: str | None = None) -> pd.DataFrame:
        """
        Collect trending tweets & engagement data.
        Uses Tweepy if credentials available, otherwise simulates.
        """
        if TWITTER_BEARER_TOKEN:
            return self._collect_real(date)
        return self._simulate(date)

    def _collect_real(self, date: str | None = None) -> pd.DataFrame:
        """Real Twitter API collection via Tweepy."""
        try:
            import tweepy
            client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
            # Fetch recent trending tweets
            query = "trending OR viral -is:retweet lang:en"
            tweets = client.search_recent_tweets(
                query=query, max_results=100,
                tweet_fields=["public_metrics", "created_at", "lang", "author_id"],
            )
            records = []
            if tweets.data:
                for tweet in tweets.data:
                    metrics = tweet.public_metrics or {}
                    records.append({
                        "post_id": str(tweet.id),
                        "caption": tweet.text,
                        "platform": "Twitter",
                        "views": metrics.get("impression_count", 0),
                        "likes": metrics.get("like_count", 0),
                        "comments": metrics.get("reply_count", 0),
                        "shares": metrics.get("retweet_count", 0),
                        "posting_timestamp": tweet.created_at,
                        "language": tweet.lang or "en",
                    })
            df = pd.DataFrame(records)
            return self.validate(df)
        except Exception as e:
            logger.warning(f"Twitter API error: {e}. Using simulation.")
            return self._simulate(date)

    def _simulate(self, date: str | None = None, n: int = 500) -> pd.DataFrame:
        """Simulate Twitter daily report."""
        rng = np.random.default_rng(hash(date or "today") % 2**31)
        return self._generate_simulated(rng, n, "Twitter")

    @staticmethod
    def _generate_simulated(rng, n, platform):
        followers = rng.lognormal(7, 2, n).astype(int)
        engagement = rng.beta(2, 20, n)
        views = (followers * engagement * rng.lognormal(0, 0.8, n)).astype(int)
        return pd.DataFrame({
            "post_id": [f"{platform.lower()}_{i}" for i in range(n)],
            "caption": [f"Simulated {platform} post #{i}" for i in range(n)],
            "platform": platform,
            "country": rng.choice(COUNTRIES, n),
            "language": rng.choice(LANGUAGES, n),
            "category": rng.choice(CATEGORIES, n),
            "content_type": rng.choice(CONTENT_TYPES, n),
            "follower_count": followers,
            "hist_engagement_rate": engagement.round(6),
            "views": views,
            "likes": (views * rng.beta(2, 30, n)).astype(int),
            "comments": (views * rng.beta(1.5, 80, n)).astype(int),
            "shares": (views * rng.beta(1.2, 100, n)).astype(int),
            "virality_class": [virality_label(v, VIRALITY_THRESHOLDS) for v in views],
            "posting_timestamp": pd.Timestamp.now(),
        })


# ═══════════════════════════════════════════════════════════════════════
#  YOUTUBE COLLECTOR
# ═══════════════════════════════════════════════════════════════════════
class YouTubeCollector(BaseCollector):
    platform = "YouTube"

    def collect(self, date: str | None = None) -> pd.DataFrame:
        if YOUTUBE_API_KEY:
            return self._collect_real(date)
        return self._simulate(date)

    def _collect_real(self, date: str | None = None) -> pd.DataFrame:
        try:
            from googleapiclient.discovery import build
            youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
            request = youtube.videos().list(
                part="snippet,statistics",
                chart="mostPopular",
                regionCode="US",
                maxResults=50,
            )
            response = request.execute()
            records = []
            for item in response.get("items", []):
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                records.append({
                    "post_id": item["id"],
                    "caption": snippet.get("title", ""),
                    "description": snippet.get("description", "")[:500],
                    "platform": "YouTube",
                    "category": snippet.get("categoryId", "Unknown"),
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "comments": int(stats.get("commentCount", 0)),
                    "posting_timestamp": snippet.get("publishedAt"),
                    "language": snippet.get("defaultLanguage", "en"),
                })
            return self.validate(pd.DataFrame(records))
        except Exception as e:
            logger.warning(f"YouTube API error: {e}. Using simulation.")
            return self._simulate(date)

    def _simulate(self, date: str | None = None, n: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(hash(date or "today") % 2**31 + 1)
        return TwitterCollector._generate_simulated(rng, n, "YouTube")


# ═══════════════════════════════════════════════════════════════════════
#  INSTAGRAM COLLECTOR
# ═══════════════════════════════════════════════════════════════════════
class InstagramCollector(BaseCollector):
    platform = "Instagram"

    def collect(self, date: str | None = None) -> pd.DataFrame:
        return self._simulate(date)

    def _simulate(self, date: str | None = None, n: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(hash(date or "today") % 2**31 + 2)
        return TwitterCollector._generate_simulated(rng, n, "Instagram")


# ═══════════════════════════════════════════════════════════════════════
#  TIKTOK COLLECTOR
# ═══════════════════════════════════════════════════════════════════════
class TikTokCollector(BaseCollector):
    platform = "TikTok"

    def collect(self, date: str | None = None) -> pd.DataFrame:
        return self._simulate(date)

    def _simulate(self, date: str | None = None, n: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(hash(date or "today") % 2**31 + 3)
        return TwitterCollector._generate_simulated(rng, n, "TikTok")


# ═══════════════════════════════════════════════════════════════════════
#  AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════
class DailyDataAggregator:
    """Collect from all platforms, validate, and save daily report."""

    def __init__(self):
        self.collectors = [
            TwitterCollector(),
            YouTubeCollector(),
            InstagramCollector(),
            TikTokCollector(),
        ]

    def collect_daily(self, date: str | None = None) -> pd.DataFrame:
        """Collect and aggregate data from all platforms."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Collecting daily data for {date} …")
        frames = []
        for collector in self.collectors:
            try:
                df = collector.collect(date)
                df = collector.validate(df)
                frames.append(df)
                logger.info(f"  {collector.platform}: {len(df)} records")
            except Exception as e:
                logger.error(f"  {collector.platform} failed: {e}")

        if not frames:
            logger.warning("No data collected!")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined["collection_date"] = date

        # Add missing columns with defaults
        for col in ["hashtags", "description", "posting_hour", "posting_day",
                     "account_age_days", "is_verified", "has_image", "has_video",
                     "has_cta", "has_url", "is_reply", "mentions_count",
                     "emoji_count", "sentiment", "prev_avg_views",
                     "caption_length", "n_hashtags", "saves",
                     "text_emb_mean", "img_emb_mean"]:
            if col not in combined.columns:
                if col in ("hashtags", "description"):
                    combined[col] = ""
                elif col in ("text_emb_mean", "img_emb_mean", "sentiment", "hist_engagement_rate"):
                    combined[col] = 0.0
                else:
                    combined[col] = 0

        # Save
        out_path = RAW_DATA_DIR / f"daily_feed_{date}.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info(f"Daily report saved: {out_path} ({len(combined)} rows)")

        return combined


# ═══════════════════════════════════════════════════════════════════════
#  TREND ANALYZER
# ═══════════════════════════════════════════════════════════════════════
class TrendAnalyzer:
    """Analyze trends from collected data."""

    @staticmethod
    def platform_trends(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate metrics by platform."""
        return df.groupby("platform").agg(
            total_posts=("post_id", "count"),
            avg_views=("views", "mean"),
            median_views=("views", "median"),
            total_likes=("likes", "sum"),
            avg_engagement=("hist_engagement_rate", "mean"),
        ).round(2).sort_values("avg_views", ascending=False)

    @staticmethod
    def category_trends(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate metrics by category."""
        return df.groupby("category").agg(
            total_posts=("post_id", "count"),
            avg_views=("views", "mean"),
            virality_rate=("virality_class", lambda x: (x == "High").mean()),
        ).round(4).sort_values("avg_views", ascending=False)

    @staticmethod
    def country_trends(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate metrics by country."""
        return df.groupby("country").agg(
            total_posts=("post_id", "count"),
            avg_views=("views", "mean"),
        ).round(2).sort_values("avg_views", ascending=False)

    @staticmethod
    def hourly_trends(df: pd.DataFrame) -> pd.DataFrame:
        """Best posting hours."""
        if "posting_hour" not in df.columns:
            return pd.DataFrame()
        return df.groupby("posting_hour").agg(
            avg_views=("views", "mean"),
        ).round(2).sort_values("avg_views", ascending=False)
