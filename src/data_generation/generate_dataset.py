"""
Synthetic Dataset Generator — 1,000,000+ realistic social media post records.

Each row simulates a social media post with:
  - Text features: caption, hashtags, description
  - Tabular features: platform, posting time, historical engagement,
    country, language, content type, category, follower count, etc.
  - Target: views/reach (regression) & virality class (classification)

Image and video embeddings are simulated as random vectors drawn from
distributions that correlate with the target, making the dataset
realistic enough for model training.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CATEGORIES, CONTENT_TYPES, COUNTRIES, DATASET_FILENAME,
    DATASET_SIZE, IMAGE_EMBEDDING_DIM, LANGUAGES, PLATFORMS,
    RAW_DATA_DIR, TEXT_EMBEDDING_DIM, VIRALITY_THRESHOLDS,
)
from src.utils.helpers import get_logger, seed_everything, virality_label

logger = get_logger("dataset_generator")

# ── Caption templates per category ───────────────────────────────────
_CAPTION_TEMPLATES = {
    "Entertainment": [
        "Just watched the most incredible show ever 🎬 #entertainment #viral",
        "This is pure gold 😂🔥 Who else saw this? #trending #funny",
        "Movie night hits different when you have popcorn 🍿 #movies",
        "Can't stop laughing at this 😂 Share with your friends! #comedy",
        "This scene literally blew my mind 🤯 #mustwatch #amazing",
    ],
    "Education": [
        "5 things I wish I knew earlier about {topic} 📚 #education #learn",
        "Study with me 📖 Productive morning routine! #studygram #motivation",
        "Mind-blowing science fact you didn't know 🧪 #science #facts",
        "Here's how to master {topic} in 30 days ⏳ #learning #tips",
        "Free resources to learn {topic} from scratch 🚀 #education #free",
    ],
    "Technology": [
        "This AI tool just changed everything 🤖 #tech #ai #innovation",
        "New gadget review — is it worth the hype? 📱 #techreview #gadgets",
        "Breaking: major update from {company} 💻 #technews #update",
        "Top 10 apps you NEED in 2025 📲 #apps #productivity",
        "The future of tech is here 🔮 #innovation #future",
    ],
    "Sports": [
        "What a game last night! 🏆 #sports #championship #winners",
        "Training day — no excuses 💪 #fitness #grind #sports",
        "GOAT debate settled once and for all 🐐 #sports #legends",
        "Incredible play! Did you see that? ⚽🏀 #highlights",
        "Pre-game rituals 🏟 Ready to dominate #gameday",
    ],
    "Music": [
        "New track dropping midnight 🎵🔥 #newmusic #release",
        "This song has been on repeat all day 🎧 #vibes #playlist",
        "Cover of {song} — let me know what you think! 🎤 #music",
        "Studio session went crazy today 🎹 #producer #beats",
        "Festival season is coming! 🎪 #musicfestival #concert",
    ],
    "Gaming": [
        "Clutched it in the last second! 🎮 #gaming #victory #esports",
        "New game review — should you buy it? 🕹 #gamereview",
        "Stream highlight from last night 📺 #twitch #streaming",
        "This glitch is absolutely insane 😱 #gamingclips #bug",
        "My setup tour 2025 💻🖥 #gamingsetup #pcgaming",
    ],
    "Fashion": [
        "Outfit of the day ✨ #fashion #ootd #style",
        "Thrift haul — look what I found! 🛍 #thrift #sustainable",
        "Summer collection vibes 🌴👗 #fashionweek #trends",
        "Style tip: less is more 🖤 #minimalist #fashion",
        "Behind the scenes of our photoshoot 📸 #model #bts",
    ],
    "Food": [
        "Made this from scratch and it's AMAZING 🍝 #foodie #recipe",
        "Best restaurant in town — you NEED to try this 🍔 #foodreview",
        "Quick & healthy meal prep ideas 🥗 #mealprep #healthy",
        "Grandma's secret recipe finally revealed 👵🍰 #cooking",
        "Food challenge: eating the spiciest dish 🌶🔥 #challenge",
    ],
    "Travel": [
        "Hidden gem you've never heard of 🗺 #travel #explore #wanderlust",
        "Sunset over the mountains 🌄 #nature #beautiful",
        "Budget travel guide: {destination} on $50/day 💰 #budget",
        "Passport stamp collection growing 🛂 #travelgoals",
        "The most beautiful place I've ever visited 😍 #paradise",
    ],
    "Health": [
        "Morning routine for mental clarity ☀️ #health #wellness #mindset",
        "5 habits that changed my life 🧘 #healthylifestyle",
        "Workout of the day — let's get it 💪 #fitness #gym",
        "Nutrition facts you need to know 🥦 #nutrition #health",
        "Self-care Sunday 🛁 #selfcare #mentalhealth",
    ],
    "Finance": [
        "How I saved $10K in 6 months 💸 #finance #saving #money",
        "Investing 101: start with these steps 📈 #investing #wealth",
        "Crypto analysis for this week 🪙 #crypto #blockchain",
        "5 passive income ideas for 2025 💰 #income #sidehustle",
        "Budget tips that actually work 📊 #budgeting #finance",
    ],
    "News": [
        "BREAKING: major development in {event} 📰 #news #breaking",
        "Here's what you need to know today 🗞 #dailynews #update",
        "Analysis: what this means for the future 🔍 #analysis",
        "Opinion: the real story behind {event} 💭 #opinion #news",
        "Weekly wrap-up: top stories you missed 📋 #recap",
    ],
    "Comedy": [
        "I can't believe I did this 😂💀 #funny #comedy #lol",
        "POV: when life gives you lemons 🍋😂 #relatable #humor",
        "Tag someone who does this 👇😂 #comedy #memes",
        "Stand-up clip — wait for the punchline 🎤 #standup",
        "My reaction when I see my bank account 💀 #broke #funny",
    ],
    "Science": [
        "This discovery changes everything we know 🔬 #science #research",
        "Space fact of the day 🚀🌌 #astronomy #space",
        "How does {phenomenon} actually work? 🧠 #explained #science",
        "New study reveals surprising findings 📊 #study #data",
        "The chemistry behind everyday things ⚗️ #chemistry",
    ],
    "Art": [
        "Work in progress — what do you think? 🎨 #art #creative",
        "Digital art timelapse 🖌 #illustration #artprocess",
        "Street art from around the world 🌍 #graffiti #urbanart",
        "Commission piece complete! 🖼 #artist #commission",
        "Art challenge day {n} 🎭 #artchallenge #drawing",
    ],
    "Politics": [
        "What just happened in {place}? 🏛 #politics #government",
        "Policy analysis: impact on citizens 📜 #policy #analysis",
        "Election update: latest polls show... 🗳 #election #vote",
        "Debate recap: key moments 🎙 #debate #politics",
        "Opinion: what needs to change now ✊ #reform",
    ],
    "Lifestyle": [
        "My morning routine for a productive day ☀️ #lifestyle #routine",
        "Home makeover on a budget 🏠 #homedecor #diy",
        "Minimalist living: one month update 📦 #minimalism",
        "Day in my life vlog 📹 #dayinmylife #vlog",
        "Life hacks that save you hours ⏰ #lifehacks #tips",
    ],
    "Motivation": [
        "You are capable of more than you think 💪 #motivation #mindset",
        "Success story: from zero to hero 🚀 #inspiration #success",
        "Daily reminder: progress > perfection 📈 #growth",
        "This quote changed my perspective 💭 #quotes #wisdom",
        "Monday motivation — let's crush this week! 🔥 #mondaymotivation",
    ],
    "Beauty": [
        "Get ready with me ✨ #grwm #beauty #makeup",
        "Skincare routine that actually works 🧴 #skincare #glow",
        "Honest product review — is it worth it? 💄 #beautyreview",
        "Hair transformation 💇‍♀️ #hairstyle #beforeandafter",
        "Natural beauty tips 🌿 #natural #beautytips",
    ],
    "Pets": [
        "Meet my new best friend 🐶❤️ #pets #puppy #cute",
        "Cat being dramatic as usual 😹 #cats #funny #catsofinstagram",
        "Pet care tips every owner should know 🐾 #petcare",
        "Rescue story — adopted this angel 🥺 #adoptdontshop",
        "Animals being derpy compilation 🤣 #funnyanimals",
    ],
}

_HASHTAG_POOL = [
    "#viral", "#trending", "#fyp", "#foryou", "#explore", "#reels",
    "#instadaily", "#photooftheday", "#love", "#instagood", "#cute",
    "#happy", "#fun", "#like4like", "#follow", "#share", "#comment",
    "#giveaway", "#motivation", "#goals", "#blessed", "#mood",
    "#vibes", "#aesthetic", "#contentcreator", "#influencer",
    "#growthmindset", "#digitalmarketing", "#socialmedia", "#brand",
]

_TOPICS = ["AI", "math", "history", "programming", "design"]
_COMPANIES = ["Apple", "Google", "Tesla", "Meta", "Microsoft"]
_SONGS = ["Shape of You", "Blinding Lights", "Bad Guy", "Bohemian Rhapsody"]
_DESTINATIONS = ["Bali", "Tokyo", "Paris", "Iceland", "Patagonia"]
_EVENTS = ["the summit", "the new policy", "the crisis"]
_PLACES = ["Washington", "Brussels", "Delhi", "London"]


def _random_caption(category: str, rng: np.random.Generator) -> str:
    """Generate a random caption for a category."""
    templates = _CAPTION_TEMPLATES.get(category, _CAPTION_TEMPLATES["Entertainment"])
    template = templates[rng.integers(0, len(templates))]
    # Fill placeholders
    template = template.replace("{topic}", rng.choice(_TOPICS))
    template = template.replace("{company}", rng.choice(_COMPANIES))
    template = template.replace("{song}", rng.choice(_SONGS))
    template = template.replace("{destination}", rng.choice(_DESTINATIONS))
    template = template.replace("{event}", rng.choice(_EVENTS))
    template = template.replace("{place}", rng.choice(_PLACES))
    template = template.replace("{n}", str(rng.integers(1, 31)))
    template = template.replace("{phenomenon}", rng.choice(["magnets", "rainbows", "black holes"]))
    return template


def _random_hashtags(rng: np.random.Generator, n_extra: int = 3) -> str:
    """Generate random extra hashtags."""
    chosen = rng.choice(_HASHTAG_POOL, size=min(n_extra, len(_HASHTAG_POOL)), replace=False)
    return " ".join(chosen)


def generate_dataset(
    n_rows: int = DATASET_SIZE,
    seed: int = 42,
    save: bool = True,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """
    Generate a synthetic social media virality dataset.

    The generation uses correlated distributions so that features
    are meaningfully related to the target (views), enabling a model
    to learn realistic patterns.
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    logger.info(f"Generating {n_rows:,} synthetic rows (chunk_size={chunk_size:,}) …")

    all_chunks = []
    n_chunks = (n_rows + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(n_chunks), desc="Generating chunks"):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_rows)
        size = end - start

        # ── Platform ─────────────────────────────────────────────
        platform = rng.choice(PLATFORMS, size=size)

        # Platform multiplier (TikTok & YouTube get more views on average)
        platform_mult = np.ones(size)
        for i, p in enumerate(platform):
            if p == "TikTok":
                platform_mult[i] = 2.5
            elif p == "YouTube":
                platform_mult[i] = 2.0
            elif p == "Instagram":
                platform_mult[i] = 1.5
            elif p == "Twitter":
                platform_mult[i] = 1.0
            elif p == "Facebook":
                platform_mult[i] = 0.9
            elif p == "LinkedIn":
                platform_mult[i] = 0.6
            elif p == "Reddit":
                platform_mult[i] = 0.8

        # ── Country & Language ───────────────────────────────────
        country = rng.choice(COUNTRIES, size=size)
        language = rng.choice(LANGUAGES, size=size)

        # ── Content type & category ──────────────────────────────
        content_type = rng.choice(CONTENT_TYPES, size=size)
        category = rng.choice(CATEGORIES, size=size)

        # Content-type multiplier
        ctype_mult = np.ones(size)
        for i, ct in enumerate(content_type):
            if ct in ("reel", "video"):
                ctype_mult[i] = 1.8
            elif ct in ("carousel", "story"):
                ctype_mult[i] = 1.2
            elif ct == "live":
                ctype_mult[i] = 1.5

        # ── Follower count (log-normal) ──────────────────────────
        follower_count = np.clip(
            rng.lognormal(mean=8.0, sigma=2.5, size=size), 10, 500_000_000
        ).astype(int)

        # ── Historical engagement rate ───────────────────────────
        hist_engagement_rate = np.clip(
            rng.beta(2, 20, size=size), 0.001, 0.40
        )

        # ── Posting hour (0-23) — peak hours get boost ───────────
        posting_hour = rng.integers(0, 24, size=size)
        hour_mult = np.where(
            (posting_hour >= 18) | (posting_hour <= 10),
            1.3, 0.85
        )

        # ── Day of week ──────────────────────────────────────────
        posting_day = rng.integers(0, 7, size=size)  # 0=Mon … 6=Sun
        day_mult = np.where(posting_day >= 5, 1.2, 1.0)  # weekends boost

        # ── Account age (days) ───────────────────────────────────
        account_age_days = rng.integers(1, 5000, size=size)

        # ── Number of hashtags ───────────────────────────────────
        n_hashtags = rng.integers(0, 31, size=size)
        hashtag_mult = np.clip(1.0 + 0.02 * n_hashtags, 1.0, 1.5)

        # ── Caption length ───────────────────────────────────────
        caption_length = rng.integers(10, 2200, size=size)

        # ── Has media? ───────────────────────────────────────────
        has_image = rng.choice([0, 1], size=size, p=[0.15, 0.85])
        has_video = np.where(
            np.isin(content_type, ["video", "reel", "live", "story"]), 1, 0
        )

        # ── Is verified ─────────────────────────────────────────
        is_verified = rng.choice(
            [0, 1], size=size, p=[0.92, 0.08]
        )
        verified_mult = np.where(is_verified == 1, 2.0, 1.0)

        # ── Has call to action ───────────────────────────────────
        has_cta = rng.choice([0, 1], size=size, p=[0.6, 0.4])
        cta_mult = np.where(has_cta == 1, 1.15, 1.0)

        # ── Mentions count ───────────────────────────────────────
        mentions_count = rng.integers(0, 11, size=size)

        # ── Emoji count ──────────────────────────────────────────
        emoji_count = rng.integers(0, 21, size=size)

        # ── URL included ─────────────────────────────────────────
        has_url = rng.choice([0, 1], size=size, p=[0.7, 0.3])

        # ── Is reply / quote / repost ────────────────────────────
        is_reply = rng.choice([0, 1], size=size, p=[0.8, 0.2])

        # ── Previous post avg views ──────────────────────────────
        prev_avg_views = np.clip(
            follower_count * hist_engagement_rate * rng.uniform(0.5, 2.0, size=size),
            0, 100_000_000
        ).astype(int)

        # ── Sentiment score (-1 to 1) ────────────────────────────
        sentiment = np.clip(rng.normal(0.2, 0.4, size=size), -1, 1)
        sentiment_mult = 1.0 + 0.2 * sentiment   # positive sentiment boosts

        # ═══════════════════════════════════════════════════════════
        #  TARGET: Views / Reach
        # ═══════════════════════════════════════════════════════════
        base_views = (
            follower_count
            * hist_engagement_rate
            * platform_mult
            * ctype_mult
            * hour_mult
            * day_mult
            * hashtag_mult
            * verified_mult
            * cta_mult
            * sentiment_mult
        )
        # Add noise
        noise = rng.lognormal(mean=0, sigma=0.8, size=size)
        views = np.clip(base_views * noise, 0, 2_000_000_000).astype(int)

        # ── Derived engagement ───────────────────────────────────
        like_rate = np.clip(rng.beta(2, 30, size=size), 0.001, 0.2)
        likes = (views * like_rate).astype(int)

        comment_rate = np.clip(rng.beta(1.5, 80, size=size), 0.0001, 0.05)
        comments = (views * comment_rate).astype(int)

        share_rate = np.clip(rng.beta(1.2, 100, size=size), 0.00005, 0.03)
        shares = (views * share_rate).astype(int)

        save_rate = np.clip(rng.beta(1.3, 120, size=size), 0.00003, 0.02)
        saves = (views * save_rate).astype(int)

        # ── Virality class ───────────────────────────────────────
        virality_class = [virality_label(v, VIRALITY_THRESHOLDS) for v in views]

        # ── Captions & hashtags ──────────────────────────────────
        captions = [_random_caption(cat, rng) for cat in category]
        extra_hashtags = [_random_hashtags(rng, int(nh)) for nh in n_hashtags]
        descriptions = [
            f"Posted by creator on {p}. Category: {cat}."
            for p, cat in zip(platform, category)
        ]

        # ── Simulated text embedding (mean vector per row) ───────
        text_emb_mean = (
            np.log1p(views) / 25.0
            + rng.normal(0, 0.1, size=size)
        )

        # ── Simulated image embedding (mean vector per row) ──────
        img_emb_mean = (
            np.log1p(views) / 30.0
            + rng.normal(0, 0.08, size=size)
        )

        # ── Posting timestamp (random date in last 3 years) ──────
        days_ago = rng.integers(0, 1095, size=size)
        hours = rng.integers(0, 24, size=size)
        minutes = rng.integers(0, 60, size=size)
        posting_timestamp = pd.Timestamp("2025-12-31") - pd.to_timedelta(
            days_ago * 86400 + hours * 3600 + minutes * 60, unit="s"
        )
        # Convert to array of timestamps (vectorised)
        base_ts = pd.Timestamp("2025-12-31")
        offsets = pd.to_timedelta(
            days_ago.astype("int64") * 86400
            + hours.astype("int64") * 3600
            + minutes.astype("int64") * 60,
            unit="s",
        )
        posting_timestamps = base_ts - offsets

        # ── Build DataFrame ──────────────────────────────────────
        chunk_df = pd.DataFrame({
            # Identifiers
            "post_id": np.arange(start, end),
            "posting_timestamp": posting_timestamps,

            # Text features
            "caption": captions,
            "hashtags": extra_hashtags,
            "description": descriptions,
            "caption_length": caption_length,
            "n_hashtags": n_hashtags,

            # Tabular features
            "platform": platform,
            "country": country,
            "language": language,
            "content_type": content_type,
            "category": category,
            "follower_count": follower_count,
            "hist_engagement_rate": np.round(hist_engagement_rate, 6),
            "posting_hour": posting_hour,
            "posting_day": posting_day,
            "account_age_days": account_age_days,
            "is_verified": is_verified,
            "has_image": has_image,
            "has_video": has_video,
            "has_cta": has_cta,
            "has_url": has_url,
            "is_reply": is_reply,
            "mentions_count": mentions_count,
            "emoji_count": emoji_count,
            "sentiment": np.round(sentiment, 4),
            "prev_avg_views": prev_avg_views,

            # Simulated embedding summaries (mean of embedding vecs)
            "text_emb_mean": np.round(text_emb_mean, 6),
            "img_emb_mean": np.round(img_emb_mean, 6),

            # Engagement (derived from views — for analysis/charts)
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "saves": saves,

            # Targets
            "views": views,
            "virality_class": virality_class,
        })

        all_chunks.append(chunk_df)

    df = pd.concat(all_chunks, ignore_index=True)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Virality distribution:\n{df['virality_class'].value_counts()}")

    if save:
        out_path = RAW_DATA_DIR / DATASET_FILENAME
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"Saved to {out_path}")

    return df


# ─── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic virality dataset")
    parser.add_argument("--rows", type=int, default=DATASET_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(n_rows=args.rows, seed=args.seed)
