import os
import re
import pandas as pd
import spacy
from tqdm import tqdm
import sys

# Add project root to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPACY_MODEL,
    PHONE_FEATURES
)

# Load spaCy model
try:
    nlp = spacy.load(SPACY_MODEL)
    print(f"Loaded spaCy model: {SPACY_MODEL}")
except OSError:
    print(f"Downloading spaCy model: {SPACY_MODEL}")
    spacy.cli.download(SPACY_MODEL)
    nlp = spacy.load(SPACY_MODEL)


def clean_text(text):
    """
    Cleans text by removing HTML tags, URLs, and special characters.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove user mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def normalize_phone_models(text):
    """
    Normalizes phone model names in text.

    Args:
        text: Input text string

    Returns:
        Text with normalized phone model references
    """
    # Example normalization patterns (expand for more models)
    normalization_patterns = [
        # iPhone variants
        (r'(?i)iphone\s*15\s*pro\s*max', 'iPhone 15 Pro Max'),
        (r'(?i)iphone\s*15\s*pro', 'iPhone 15 Pro'),
        (r'(?i)iphone\s*15\s*plus', 'iPhone 15 Plus'),
        (r'(?i)iphone\s*15', 'iPhone 15'),

        # Samsung variants
        (r'(?i)s23\s*ultra', 'Samsung Galaxy S23 Ultra'),
        (r'(?i)s23\s*\+', 'Samsung Galaxy S23+'),
        (r'(?i)s23', 'Samsung Galaxy S23'),
        (r'(?i)samsung\s*s23\s*ultra', 'Samsung Galaxy S23 Ultra'),

        # Google Pixel variants
        (r'(?i)pixel\s*8\s*pro', 'Google Pixel 8 Pro'),
        (r'(?i)pixel\s*8', 'Google Pixel 8'),

        # OnePlus variants
        (r'(?i)oneplus\s*12', 'OnePlus 12'),
        (r'(?i)1\+\s*12', 'OnePlus 12')
    ]

    for pattern, replacement in normalization_patterns:
        text = re.sub(pattern, replacement, text)

    return text


def normalize_technical_terms(text):
    """
    Standardizes technical specifications and units.

    Args:
        text: Input text string

    Returns:
        Text with normalized technical terms
    """
    # Battery capacity normalization
    text = re.sub(r'(\d+)\s*mah', r'\1mAh', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)mah', r'\1mAh', text, flags=re.IGNORECASE)

    # Screen refresh rate
    text = re.sub(r'(\d+)\s*hz', r'\1Hz', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)hz', r'\1Hz', text, flags=re.IGNORECASE)

    # Screen resolution
    text = re.sub(r'(\d+)\s*[xX]\s*(\d+)', r'\1x\2', text)

    # Camera megapixels
    text = re.sub(r'(\d+)\s*mp', r'\1MP', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)mp', r'\1MP', text, flags=re.IGNORECASE)

    # Storage capacity
    text = re.sub(r'(\d+)\s*gb', r'\1GB', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)gb', r'\1GB', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*tb', r'\1TB', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)tb', r'\1TB', text, flags=re.IGNORECASE)

    # RAM
    text = re.sub(r'(\d+)\s*gb\s*ram', r'\1GB RAM', text, flags=re.IGNORECASE)

    return text


def identify_mentioned_features(text, feature_dict=PHONE_FEATURES):
    """
    Identifies which phone features are mentioned in the text.

    Args:
        text: Input text string
        feature_dict: Dictionary of features and related terms

    Returns:
        Dictionary with features as keys and boolean values indicating presence
    """
    text_lower = text.lower()
    mentioned_features = {}

    for feature, keywords in feature_dict.items():
        # Check for the main feature name
        if feature in text_lower:
            mentioned_features[feature] = True
            continue

        # Check for related keywords
        for keyword in keywords:
            if keyword in text_lower:
                mentioned_features[feature] = True
                break

        # If feature is not found, set to False
        if feature not in mentioned_features:
            mentioned_features[feature] = False

    return mentioned_features


def preprocess_comments(phone_model=None):
    """
    Preprocesses comments for a specific phone model or all available data.

    Args:
        phone_model: Optional phone model name. If None, processes all files.

    Returns:
        DataFrame with preprocessed comments
    """
    # Determine which files to process
    if phone_model:
        input_files = [os.path.join(RAW_DATA_DIR, f"{phone_model.replace(' ', '_')}_comments.csv")]
    else:
        input_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('_comments.csv')]
        input_files = [os.path.join(RAW_DATA_DIR, f) for f in input_files]

    all_processed_data = []

    for input_file in input_files:
        print(f"Processing file: {input_file}")

        # Load raw comments
        try:
            df = pd.read_csv(input_file)
            print(f"Loaded {len(df)} comments")
        except Exception as e:
            print(f"Error loading file {input_file}: {e}")
            continue

        # Basic cleaning
        df['cleaned_text'] = df['text'].apply(clean_text)

        # Filter out short or empty comments
        df = df[df['cleaned_text'].str.len() > 10].reset_index(drop=True)
        print(f"After filtering short comments: {len(df)} comments remaining")

        # Normalize phone model references
        df['normalized_text'] = df['cleaned_text'].apply(normalize_phone_models)

        # Normalize technical terms
        df['normalized_text'] = df['normalized_text'].apply(normalize_technical_terms)