"""
Text preprocessing utilities for the mobile phone sentiment analysis system.
"""

import re
import nltk
import spacy
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import logging
import sys
import string
import random

# Import configuration
sys.path.append(".")
from mobile_phone_sentiment_analysis.config.config import NLTK_RESOURCES, SPACY_MODEL, TEXT_PREPROCESSING

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_nltk():
    """Download and initialize required NLTK resources."""
    for resource in NLTK_RESOURCES:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.error(f"Failed to download NLTK resource {resource}: {str(e)}")
            
def initialize_spacy():
    """Initialize SpaCy model."""
    try:
        nlp = spacy.load(SPACY_MODEL)
        logger.info(f"Successfully loaded SpaCy model: {SPACY_MODEL}")
        return nlp
    except Exception as e:
        logger.error(f"Failed to load SpaCy model {SPACY_MODEL}: {str(e)}")
        logger.info("Attempting to download the model...")
        try:
            spacy.cli.download(SPACY_MODEL)
            nlp = spacy.load(SPACY_MODEL)
            logger.info(f"Successfully downloaded and loaded SpaCy model: {SPACY_MODEL}")
            return nlp
        except Exception as e2:
            logger.error(f"Failed to download SpaCy model {SPACY_MODEL}: {str(e2)}")
            return None

def clean_text(text):
    """
    Basic text cleaning.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and numbers (keep letters, spaces, and basic punctuation for sentence structure)
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_synonyms(word):
    """Get synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def synonym_replacement(text, n=1):
    """Replace n words in the text with their synonyms."""
    words = text.split()
    if len(words) <= 1:
        return text
    
    # Only consider words with 3+ chars that aren't stopwords
    stop_words = set(stopwords.words('english'))
    candidate_words = [word for word in words if len(word) >= 3 and word not in stop_words]
    
    if not candidate_words:
        return text
    
    n = min(n, len(candidate_words))
    replace_indices = random.sample(range(len(candidate_words)), n)
    
    new_words = words.copy()
    for i in replace_indices:
        word = candidate_words[i]
        word_idx = words.index(word)
        
        synonyms = get_synonyms(word)
        if synonyms:
            new_words[word_idx] = random.choice(synonyms)
    
    return ' '.join(new_words)

def random_swap(text, n=1):
    """Randomly swap n pairs of words in the text."""
    words = text.split()
    if len(words) <= 1:
        return text
    
    n = min(n, len(words)//2)
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    
    return ' '.join(words)

def expand_contractions(text):
    """Expand contractions in text."""
    contractions = {
        "ain't": "is not", "aren't": "are not", "can't": "cannot", 
        "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
        "don't": "do not", "hadn't": "had not", "hasn't": "has not",
        "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
        "i've": "i have", "isn't": "is not", "it's": "it is", "let's": "let us",
        "shouldn't": "should not", "that's": "that is", "there's": "there is",
        "they'd": "they would", "they'll": "they will", "they're": "they are",
        "they've": "they have", "wasn't": "was not", "we'd": "we would",
        "we're": "we are", "we've": "we have", "weren't": "were not",
        "what's": "what is", "where's": "where is", "who's": "who is",
        "won't": "will not", "wouldn't": "would not", "you'd": "you would",
        "you'll": "you will", "you're": "you are", "you've": "you have"
    }
    
    # Sort contractions by length (longest first) to avoid partial replacements
    sorted_contractions = sorted(contractions.items(), key=lambda x: len(x[0]), reverse=True)
    
    for contraction, expansion in sorted_contractions:
        text = text.replace(contraction, expansion)
    
    return text

def preprocess_text(text, advanced=True):
    """
    Preprocess text for sentiment analysis.
    
    Args:
        text (str): Input text
        advanced (bool): Whether to use advanced preprocessing
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Basic preprocessing
    # Convert to lowercase
    if TEXT_PREPROCESSING.get('lowercase', True):
        text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emojis and special characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Special handling for negations (don't, isn't, etc.) - replace with dont, isnt
    # This preserves the negation which is important for sentiment
    text = re.sub(r"n't", "nt", text)
    
    # Expand contractions
    if advanced and TEXT_PREPROCESSING.get('expand_contractions', True):
        text = expand_contractions(text)
    
    # Remove punctuation
    if TEXT_PREPROCESSING.get('remove_punctuation', True):
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
    
    # Advanced preprocessing (optional)
    if advanced:
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if TEXT_PREPROCESSING.get('remove_stopwords', True):
            stop_words = set(stopwords.words('english'))
            # Don't remove negation words which are important for sentiment
            negations = {"no", "not", "nor", "none", "never", "neither"}
            filtered_stop_words = stop_words - negations
            tokens = [word for word in tokens if word not in filtered_stop_words]
        
        # Remove short words
        min_length = TEXT_PREPROCESSING.get('min_word_length', 2)
        tokens = [word for word in tokens if len(word) >= min_length]
        
        # Lemmatization
        if TEXT_PREPROCESSING.get('lemmatization', True):
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Stemming (usually don't use both stemming and lemmatization)
        elif TEXT_PREPROCESSING.get('stemming', False):
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
        
        # Join tokens back to string
        text = ' '.join(tokens)
    
    return text

def preprocess_with_spacy(text, nlp=None):
    """
    Preprocess text using spaCy for advanced NLP features.
    
    Args:
        text (str): Input text
        nlp: spaCy NLP object
        
    Returns:
        str: Preprocessed text
    """
    import spacy
    
    # Basic preprocessing first
    text = preprocess_text(text, advanced=False)
    
    # Load spaCy if not provided
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            # Fallback to basic preprocessing
            return preprocess_text(text, advanced=True)
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Filter tokens
    tokens = []
    for token in doc:
        # Skip stopwords, punctuation, and short words
        if token.is_stop or token.is_punct or len(token.text) < 2:
            continue
        
        # Use lemma
        tokens.append(token.lemma_)
    
    # Join tokens
    processed_text = " ".join(tokens)
    
    return processed_text

def extract_parts_of_speech(doc):
    """
    Extract parts of speech from a SpaCy document.
    
    Args:
        doc: SpaCy Doc object
        
    Returns:
        dict: Dictionary of POS-tagged tokens
    """
    if doc is None:
        return {}
    
    pos_dict = {
        'nouns': [token.text for token in doc if token.pos_ == 'NOUN'],
        'verbs': [token.text for token in doc if token.pos_ == 'VERB'],
        'adjectives': [token.text for token in doc if token.pos_ == 'ADJ'],
        'adverbs': [token.text for token in doc if token.pos_ == 'ADV']
    }
    
    return pos_dict

def extract_named_entities(doc):
    """
    Extract named entities from a SpaCy document.
    
    Args:
        doc: SpaCy Doc object
        
    Returns:
        dict: Dictionary of named entities by type
    """
    if doc is None:
        return {}
    
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    return entities

# Initialize NLTK and SpaCy when module is imported
initialize_nltk()
nlp = initialize_spacy()
