import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# YouTube API Configuration
# Note: In a real project, use environment variables for sensitive info
YOUTUBE_API_KEY = "AIzaSyCLrMoPxDQZdI6uLIYfvtFHAWsuwWpe8Y0"  # Replace with your actual API key or use from environment
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
MAX_COMMENTS_PER_VIDEO = 1000  # Adjust based on your needs and API quotas

# Feature Taxonomy
PHONE_FEATURES = {
    "camera": ["photo", "picture", "shot", "lens", "zoom", "night mode", "ultrawide", "portrait"],
    "battery": ["battery life", "charging", "power", "drain", "endurance", "screen on time", "sot"],
    "display": ["screen", "amoled", "lcd", "refresh rate", "brightness", "resolution", "hdr"],
    "performance": ["speed", "lag", "processor", "chip", "gaming", "multitasking", "snapdragon", "bionic"],
    "software": ["android", "ios", "oneui", "miui", "update", "bug", "feature", "os"],
    "design": ["build quality", "material", "plastic", "metal", "glass", "durability", "color", "size"],
    "audio": ["speaker", "sound", "headphone", "jack", "bluetooth", "quality", "loud", "bass"],
    "price": ["cost", "expensive", "cheap", "value", "worth", "price tag", "overpriced", "deal"],
    "connectivity": ["5g", "wifi", "bluetooth", "network", "signal", "reception", "nfc"]
}

# Model Parameters
BERT_MODEL_NAME = "bert-base-uncased"  # We'll fine-tune this for our specific task
MAX_SEQ_LENGTH = 128  # Maximum sequence length for the BERT model
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 5

# NLP Processing
SPACY_MODEL = "en_core_web_md"  # Medium-sized English model

# Sentiment Analysis
SENTIMENT_THRESHOLDS = {
    "positive": 0.6,  # Score above this is considered positive
    "negative": 0.4   # Score below this is considered negative, between is neutral
}