"""
Feature extraction utilities for the mobile phone sentiment analysis system.
"""

import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import logging
import re
import os
import numpy as np

# Import configuration
sys.path.append(".")
from mobile_phone_sentiment_analysis.config.config import (
    ASPECT_KEYWORDS, POSITIVE_WORDS, NEGATIVE_WORDS, 
    NEUTRAL_WORDS, INTENSIFIERS, NEGATORS
)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mobile-specific terms
MOBILE_POSITIVE = [
    'smooth', 'fast', 'responsive', 'crisp', 'vibrant', 'bright', 'sharp',
    'premium', 'slim', 'lightweight', 'durable', 'waterproof', 'flagship',
    'snappy', 'quick', 'fluid', 'stable', 'reliable', 'accurate', 'powerful',
    'ultra', 'pro', 'hdr', 'amoled', 'oled', '4k', '5g', '120hz', '144hz',
    'silky', 'seamless', 'immersive', 'stunning', 'gorgeous', 'impressive',
    'excellent', 'stellar', 'solid', 'buttery', 'worth', 'recommend'
]

MOBILE_NEGATIVE = [
    'lag', 'laggy', 'slow', 'sluggish', 'freeze', 'freezes', 'crash',
    'crashes', 'blurry', 'grainy', 'dim', 'dull', 'cheap', 'plasticky',
    'flimsy', 'fragile', 'heavy', 'bulky', 'thick', 'glitchy', 'unstable',
    'stutter', 'stutters', 'choppy', 'overpriced', 'expensive', 'overheating',
    'throttling', 'scratches', 'scratch', 'breaks', 'broke', 'bloatware',
    'buggy', 'mediocre', 'jitter', 'disappointing', 'drain', 'drains'
]

# Special case phrases with strong sentiment
MOBILE_PHRASES = {
    # Positive phrases
    'worth every penny': 1.0,
    'battery life is amazing': 0.9,
    'takes great photos': 0.9,
    'lasts all day': 0.8,
    'crisp display': 0.8,
    'good camera': 0.7,
    'great value': 0.8,
    'fast charging': 0.8,
    'solid performance': 0.7,
    
    # Negative phrases
    'not worth the price': -0.9,
    'battery drain': -0.8,
    'poor camera quality': -0.9,
    'overheats easily': -0.9,
    'laggy interface': -0.8,
    'too expensive': -0.8,
    'poor battery life': -0.9,
    'crappy camera': -0.9,
    'screen cracked': -0.8,
    'waste of money': -1.0
}

def extract_aspects(text, nlp=None):
    """
    Extract aspects (product features) from the text.
    
    Args:
        text (str): Input text
        nlp: SpaCy NLP model (optional)
        
    Returns:
        dict: Dictionary mapping aspects to whether they were mentioned
    """
    text_lower = text.lower()
    
    # Initialize aspects dictionary
    aspects = {
        'camera': False,
        'battery': False,
        'performance': False,
        'display': False,
        'design': False,
        'price': False
    }
    
    # Simple keyword-based approach for aspect detection
    aspect_keywords = {
        'camera': ['camera', 'photo', 'picture', 'image', 'photography', 'video', 'recording', 'selfie', 'lens', 'zoom', 'portrait', 'ultrawide', 'megapixel', 'mp'],
        'battery': ['battery', 'charging', 'power', 'drain', 'life', 'backup', 'long lasting', 'endurance', 'fast charging', 'wireless charging', 'capacity', 'mah'],
        'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'processing', 'snappy', 'responsive', 'processor', 'cpu', 'ram', 'memory', 'benchmark', 'chip', 'snapdragon', 'exynos', 'gaming', 'fps', 'multitasking'],
        'display': ['display', 'screen', 'resolution', 'bright', 'color', 'refresh rate', 'refresh', 'amoled', 'lcd', 'oled', 'panel', 'hdr', 'ppi', 'nits', 'brightness', 'contrast', 'hz', 'notch', 'bezel'],
        'design': ['design', 'build', 'feel', 'body', 'glass', 'metal', 'plastic', 'premium', 'look', 'size', 'weight', 'thickness', 'slim', 'fingerprint', 'scanner', 'face unlock', 'water resistant', 'ip68', 'gorilla glass'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'affordable', 'overpriced', 'money', 'dollar', 'dollars', '$', 'budget', 'flagship', 'premium', 'mid-range', 'high-end']
    }
    
    # Check for each aspect's keywords in the text
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                aspects[aspect] = True
                break
    
    # If no aspects are detected, try to use at least one based on content
    if not any(aspects.values()):
        # Add a default aspect if we can't detect any
        aspects['performance'] = True
    
    return aspects

def get_aspect_context(text, aspect, window_size=10):
    """
    Extract context around aspect mentions for targeted sentiment analysis.
    
    Args:
        text (str): Input text
        aspect (str): The aspect to find context for
        window_size (int): Number of words to include on each side of the aspect
        
    Returns:
        list: List of context windows containing the aspect
    """
    if not isinstance(text, str) or not text:
        return []
    
    text = text.lower()
    tokens = word_tokenize(text)
    context_windows = []
    
    # Look for aspect keywords in the text
    keywords = ASPECT_KEYWORDS.get(aspect, [])
    for keyword in keywords:
        for i, token in enumerate(tokens):
            if keyword in token:
                # Extract context window
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                context = ' '.join(tokens[start:end])
                context_windows.append(context)
    
    return context_windows

def rule_based_sentiment(text):
    """
    Use rule-based approach to determine sentiment.
    
    Args:
        text (str): Input text
        
    Returns:
        tuple: (sentiment, score) where sentiment is 'positive', 'negative', or 'neutral'
    """
    if not isinstance(text, str):
        return 'neutral', 0.0
    
    text = text.lower()
    score = 0.0
    
    # Check for mobile-specific phrases first
    for phrase, phrase_score in MOBILE_PHRASES.items():
        if phrase in text:
            # Strong phrase found, add its score with high impact
            score += phrase_score * 2
    
    # Combine general lexicons with mobile-specific ones
    all_positive = POSITIVE_WORDS + MOBILE_POSITIVE
    all_negative = NEGATIVE_WORDS + MOBILE_NEGATIVE
    
    # Split text into words
    words = re.findall(r'\b\w+\b', text)
    
    # Check for sentiment words
    for i, word in enumerate(words):
        # Initialize modifier (for intensifiers and negators)
        modifier = 1.0
        
        # Check previous words for negators and intensifiers
        for j in range(max(0, i-3), i):
            prev_word = words[j]
            if prev_word in NEGATORS:
                modifier *= -1.0
            elif prev_word in INTENSIFIERS:
                modifier *= 1.5
        
        # Calculate sentiment score
        if word in all_positive:
            score += 1.0 * modifier
        elif word in all_negative:
            score -= 1.0 * modifier
        elif word in NEUTRAL_WORDS:
            score += 0.1 * modifier  # Small positive bias for neutral words
    
    # Check for common patterns
    if 'not good' in text or 'not great' in text:
        score -= 1.0
    if 'not bad' in text or 'not terrible' in text:
        score += 0.5
    if 'could be better' in text:
        score -= 0.5
    if 'best phone' in text or 'great phone' in text:
        score += 1.5
    if 'worst phone' in text or 'terrible phone' in text:
        score -= 1.5
    
    # Normalize score by text length (to prevent long texts from having extreme scores)
    text_length = max(1, len(words))
    normalized_score = score / (text_length ** 0.5)  # Square root to dampen effect
    
    # Determine sentiment based on score
    if normalized_score > 0.1:
        sentiment = 'positive'
    elif normalized_score < -0.1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment, normalized_score

def aspect_based_sentiment(text, aspect):
    """
    Determine sentiment specifically for a given aspect.
    
    Args:
        text (str): Input text
        aspect (str): Aspect to analyze
        
    Returns:
        tuple: (sentiment, score)
    """
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Get aspect keywords
    aspect_keywords = {
        'camera': ['camera', 'photo', 'picture', 'image', 'photography', 'video', 'selfie', 'lens'],
        'battery': ['battery', 'charging', 'power', 'drain', 'life', 'backup'],
        'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'processing', 'snappy', 'processor'],
        'display': ['display', 'screen', 'resolution', 'bright', 'color', 'refresh'],
        'design': ['design', 'build', 'feel', 'body', 'glass', 'metal', 'plastic', 'premium', 'look'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'affordable', 'overpriced']
    }
    
    keywords = aspect_keywords.get(aspect, [aspect])
    
    # Find sentences that mention the aspect
    aspect_sentences = []
    for sentence in sentences:
        sentence = sentence.lower().strip()
        if not sentence:
            continue
            
        if any(keyword in sentence for keyword in keywords):
            aspect_sentences.append(sentence)
    
    # If no specific sentence mentions the aspect, fallback to entire text
    if not aspect_sentences:
        return rule_based_sentiment(text)
    
    # Analyze sentiment of each sentence that mentions the aspect
    total_score = 0.0
    for sentence in aspect_sentences:
        sentiment, score = rule_based_sentiment(sentence)
        total_score += score
    
    # Average the scores
    avg_score = total_score / len(aspect_sentences)
    
    # Determine sentiment based on average score
    if avg_score > 0.1:
        sentiment = 'positive'
    elif avg_score < -0.1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment, avg_score

def extract_common_phrases(texts, n=20, min_freq=2):
    """
    Extract common bigram phrases from a collection of texts.
    
    Args:
        texts (list): List of text strings
        n (int): Number of phrases to extract
        min_freq (int): Minimum frequency for a phrase to be included
        
    Returns:
        list: List of common phrases with their frequencies
    """
    if not texts:
        return []
    
    # Combine all texts
    all_text = ' '.join(texts)
    
    # Tokenize
    tokens = word_tokenize(all_text.lower())
    
    # Find bigram collocations
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(min_freq)
    
    # Get top phrases
    bigram_measures = BigramAssocMeasures()
    return finder.nbest(bigram_measures.pmi, n)

def extract_feature_keywords(texts, aspects):
    """
    Extract keywords associated with each aspect.
    
    Args:
        texts (list): List of text strings
        aspects (list): List of aspects to extract keywords for
        
    Returns:
        dict: Dictionary mapping aspects to most common associated words
    """
    if not texts or not aspects:
        return {}
    
    feature_keywords = {aspect: [] for aspect in aspects}
    
    for aspect in aspects:
        # Get context for this aspect
        aspect_contexts = []
        for text in texts:
            contexts = get_aspect_context(text, aspect)
            aspect_contexts.extend(contexts)
        
        if not aspect_contexts:
            continue
        
        # Extract words from contexts
        aspect_words = []
        for context in aspect_contexts:
            tokens = word_tokenize(context.lower())
            aspect_words.extend(tokens)
        
        # Count word frequencies
        word_freq = {}
        for word in aspect_words:
            if len(word) > 2 and word not in ASPECT_KEYWORDS[aspect]:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        feature_keywords[aspect] = [word for word, freq in top_words]
    
    return feature_keywords
