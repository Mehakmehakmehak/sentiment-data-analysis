"""
Aspect-specific sentiment analysis models for the mobile phone sentiment analysis system.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import glob

# Import configuration
sys.path.append(".")
from mobile_phone_sentiment_analysis.config.config import MODEL_SAVE_DIR, MAX_FEATURES, N_ESTIMATORS, RANDOM_STATE, C_VALUE, MAX_ITER, CLASS_WEIGHT, ASPECT_KEYWORDS, POSITIVE_WORDS, NEGATIVE_WORDS
from mobile_phone_sentiment_analysis.utils.preprocessing import preprocess_text, synonym_replacement

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom feature extractor for aspect-specific keywords
class AspectKeywordFeatures(BaseEstimator, TransformerMixin):
    """Extract keyword features specific to a given aspect"""
    def __init__(self, aspect_name):
        self.aspect_name = aspect_name
        self.aspect_keywords = ASPECT_KEYWORDS.get(self.aspect_name, [])
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, texts):
        features = []
        for text in texts:
            text = text.lower()
            words = text.split()
            
            # Count aspect keyword occurrences
            keyword_count = sum(1 for word in words if any(kw in word for kw in self.aspect_keywords))
            keyword_ratio = keyword_count / max(len(words), 1)
            
            # Count proximity of aspect keywords to sentiment words
            aspect_sentiment_proximity = 0
            for i, word in enumerate(words):
                if any(kw in word for kw in self.aspect_keywords):
                    # Check words before and after for sentiment words
                    context_start = max(0, i-3)
                    context_end = min(len(words), i+4)
                    context_words = words[context_start:context_end]
                    
                    pos_near = sum(1 for w in context_words if w in POSITIVE_WORDS)
                    neg_near = sum(1 for w in context_words if w in NEGATIVE_WORDS)
                    aspect_sentiment_proximity += pos_near - neg_near
            
            features.append([
                keyword_count,
                keyword_ratio,
                aspect_sentiment_proximity,
                1 if keyword_count > 0 else 0  # Binary feature for aspect presence
            ])
        
        return np.array(features)

def ensure_model_dir():
    """Ensure the model save directory exists."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    logger.info(f"Ensuring model directory exists: {MODEL_SAVE_DIR}")

class AspectSentimentModel:
    """
    A model for aspect-specific sentiment analysis.
    """
    
    def __init__(self, aspect_name):
        """
        Initialize an aspect sentiment model.
        
        Args:
            aspect_name (str): Name of the aspect this model is for
        """
        self.aspect_name = aspect_name
        self.pipeline = None
        self.trained = False
    
    def build_pipeline(self):
        """
        Build the model pipeline with preprocessing and classifier.
        """
        # Use LinearSVC with calibration to get probability estimates
        base_svm = LinearSVC(
            C=C_VALUE,
            class_weight=CLASS_WEIGHT,
            dual=False,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
            loss='squared_hinge'
        )
        
        # Wrap with CalibratedClassifierCV to get predict_proba
        calibrated_svm = CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')
        
        # Create pipeline with multiple feature types
        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('tfidf', TfidfVectorizer(
                    max_features=MAX_FEATURES,
                    min_df=2,
                    max_df=0.85,
                    sublinear_tf=True,
                    use_idf=True,
                    ngram_range=(1, 3),
                    norm='l2',
                    stop_words='english'
                )),
                ('count_vec', CountVectorizer(
                    max_features=int(MAX_FEATURES/2),
                    ngram_range=(1, 2),
                    binary=True
                )),
                ('aspect_features', AspectKeywordFeatures(self.aspect_name))
            ])),
            ('classifier', calibrated_svm)
        ])
        
        logger.info(f"Built enhanced SVM model pipeline for {self.aspect_name} aspect with aspect-specific features (C={C_VALUE}, max_iter={MAX_ITER})")
    
    def train(self, X_train, y_train):
        """
        Train the aspect sentiment model.
        
        Args:
            X_train (array-like): Training text data
            y_train (array-like): Training sentiment labels
            
        Returns:
            self: The trained model instance
        """
        if self.pipeline is None:
            self.build_pipeline()
        
        # Preprocess the text data
        X_train_processed = [preprocess_text(text) for text in X_train]
        
        # Train the model
        logger.info(f"Training {self.aspect_name} aspect model on {len(X_train_processed)} samples")
        self.pipeline.fit(X_train_processed, y_train)
        self.trained = True
        
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the aspect sentiment model.
        
        Args:
            X_test (array-like): Test text data
            y_test (array-like): Test sentiment labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.trained or self.pipeline is None:
            logger.error(f"Cannot evaluate {self.aspect_name} model - not trained yet")
            return None
        
        # Preprocess the text data
        X_test_processed = [preprocess_text(text) for text in X_test]
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test_processed)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Evaluated {self.aspect_name} aspect model - Accuracy: {accuracy:.4f}")
        
        # Return metrics
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'aspect': self.aspect_name
        }
    
    def predict(self, texts):
        """
        Predict sentiment for new texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: Predicted sentiments
        """
        if not self.trained or self.pipeline is None:
            logger.error(f"Cannot predict with {self.aspect_name} model - not trained yet")
            return None
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess the text data
        texts_processed = [preprocess_text(text) for text in texts]
        
        # Make predictions
        return self.pipeline.predict(texts_processed)
    
    def predict_proba(self, texts):
        """
        Predict sentiment probabilities for new texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            numpy.ndarray: Predicted probabilities for each class
        """
        if not self.trained or self.pipeline is None:
            logger.error(f"Cannot predict with {self.aspect_name} model - not trained yet")
            return None
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess the text data
        texts_processed = [preprocess_text(text) for text in texts]
        
        # Make predictions
        return self.pipeline.predict_proba(texts_processed)
    
    def save(self, filepath=None):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model file
            
        Returns:
            str: Path to the saved model file
        """
        if not self.trained or self.pipeline is None:
            logger.error(f"Cannot save {self.aspect_name} model - not trained yet")
            return None
        
        ensure_model_dir()
        
        if filepath is None:
            filepath = os.path.join(MODEL_SAVE_DIR, f"{self.aspect_name}_aspect_model.joblib")
        
        # Save the model
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Saved {self.aspect_name} aspect model to {filepath}")
        
        return filepath
    
    def load(self, filepath=None):
        """
        Load the model from disk.
        
        Args:
            filepath (str): Path to the model file
            
        Returns:
            self: The loaded model instance
        """
        if filepath is None:
            filepath = os.path.join(MODEL_SAVE_DIR, f"{self.aspect_name}_aspect_model.joblib")
        
        try:
            # Load the model
            self.pipeline = joblib.load(filepath)
            self.trained = True
            logger.info(f"Loaded {self.aspect_name} aspect model from {filepath}")
            return self
        except Exception as e:
            logger.error(f"Error loading {self.aspect_name} model from {filepath}: {str(e)}")
            return None

class AspectModelManager:
    """
    Manages aspect-specific sentiment models.
    """
    
    def __init__(self):
        """Initialize the aspect model manager."""
        self.models = {}
        self.trained = False
    
    def build_aspect_model(self, aspect_name):
        """
        Build an advanced model for a specific aspect.
        
        Args:
            aspect_name (str): Name of the aspect to model
            
        Returns:
            AspectSentimentModel: The built model
        """
        model = AspectSentimentModel(aspect_name)
        model.build_pipeline()
        return model
    
    def has_model(self, aspect_name):
        """Check if a model exists for the given aspect."""
        return aspect_name in self.models and self.models[aspect_name].trained
    
    def train_aspect_models(self, aspect_datasets):
        """
        Train models for all aspects in the datasets.
        
        Args:
            aspect_datasets (dict): Dictionary mapping aspects to DataFrames
            
        Returns:
            dict: Evaluation metrics for each aspect model
        """
        if aspect_datasets is None or len(aspect_datasets) == 0:
            logger.warning("No aspect datasets provided for training")
            return {}
        
        results = {}
        for aspect, df in aspect_datasets.items():
            try:
                logger.info(f"Training {aspect} aspect model...")
                
                # Check if DataFrame has required columns
                required_cols = ['comment_text', f'{aspect}_sentiment']
                if not all(col in df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in df.columns]
                    logger.error(f"Missing required columns for {aspect}: {missing}")
                    results[aspect] = {'error': f"Missing columns: {missing}"}
                    continue
                
                # Convert sentiment to string to ensure consistent types
                df[f'{aspect}_sentiment'] = df[f'{aspect}_sentiment'].astype(str)
                
                # Get sentiment distribution
                sentiment_counts = df[f'{aspect}_sentiment'].value_counts().to_dict()
                logger.info(f"{aspect.capitalize()} sentiment distribution: {sentiment_counts}")
                
                # Split data
                try:
                    # Check if we have enough examples of each class for stratification
                    min_class_count = min(sentiment_counts.values())
                    
                    if min_class_count < 2 or len(sentiment_counts) < 2:
                        # Can't use stratification if any class has <2 examples
                        logger.warning(f"Insufficient examples for stratified sampling in {aspect}. Using random split.")
                        train_df, test_df = train_test_split(
                            df, test_size=0.2, random_state=RANDOM_STATE
                        )
                    else:
                        # Use stratified sampling
                        train_df, test_df = train_test_split(
                            df, test_size=0.2, random_state=RANDOM_STATE, 
                            stratify=df[f'{aspect}_sentiment']
                        )
                except Exception as e:
                    logger.warning(f"Error in stratified split for {aspect}, using random split: {str(e)}")
                    train_df, test_df = train_test_split(
                        df, test_size=0.2, random_state=RANDOM_STATE
                    )
                
                # Apply data augmentation for minority classes
                # First identify minority classes that need augmentation
                sentiment_counts = train_df[f'{aspect}_sentiment'].value_counts()
                mean_count = sentiment_counts.mean()
                min_count = sentiment_counts.min()
                
                if min_count < mean_count * 0.5 and len(train_df) > 50:
                    logger.info(f"Applying selective augmentation for imbalanced classes in {aspect}")
                    augmented_train_df = train_df.copy()
                    
                    for sentiment, count in sentiment_counts.items():
                        if count < mean_count * 0.7:  # Target classes with <70% of the mean
                            # How many additional samples to generate
                            target_count = int(mean_count * 0.7)
                            num_to_add = min(target_count - count, count)  # Don't add more than original count
                            
                            if num_to_add > 0:
                                # Get samples of this sentiment
                                sentiment_samples = train_df[train_df[f'{aspect}_sentiment'] == sentiment]
                                
                                # Generate augmented samples
                                for _ in range(num_to_add):
                                    # Randomly select a sample to augment
                                    sample = sentiment_samples.sample(1).iloc[0]
                                    
                                    # Apply augmentation
                                    aug_text = synonym_replacement(sample['comment_text'], n=2)
                                    
                                    # Add augmented sample to training data
                                    augmented_row = sample.copy()
                                    augmented_row['comment_text'] = aug_text
                                    augmented_train_df = pd.concat([augmented_train_df, pd.DataFrame([augmented_row])])
                    
                    # Use the augmented training data
                    train_df = augmented_train_df
                    logger.info(f"After augmentation: {len(train_df)} training samples")
                
                # Create and train model
                model = self.build_aspect_model(aspect)
                model.train(
                    train_df['comment_text'],
                    train_df[f'{aspect}_sentiment']
                )
                
                # Evaluate model
                metrics = model.evaluate(
                    test_df['comment_text'],
                    test_df[f'{aspect}_sentiment']
                )
                
                # Save model
                ensure_model_dir()
                model_path = os.path.join(MODEL_SAVE_DIR, f"{aspect}_aspect_model.joblib")
                joblib.dump(model.pipeline, model_path)
                logger.info(f"Saved {aspect} aspect model to {model_path}")
                
                # Store model
                self.models[aspect] = model
                results[aspect] = metrics
                
            except Exception as e:
                logger.error(f"Error training {aspect} model: {str(e)}")
                results[aspect] = {'error': str(e)}
                continue
        
        self.trained = True
        return results
    
    def load_all_models(self):
        """Load all aspect models from disk."""
        self.models = {}
        success = True
        
        try:
            # Find all aspect models in the model directory
            model_pattern = os.path.join(MODEL_SAVE_DIR, "*_aspect_model.joblib")
            model_files = glob.glob(model_pattern)
            
            if not model_files:
                logger.warning("No aspect models found to load")
                return False
            
            # Load each model
            for model_path in model_files:
                try:
                    filename = os.path.basename(model_path)
                    aspect = filename.replace("_aspect_model.joblib", "")
                    
                    # Create a new model instance
                    model = AspectSentimentModel(aspect)
                    model.pipeline = joblib.load(model_path)
                    model.trained = True
                    
                    # Store the model
                    self.models[aspect] = model
                    logger.info(f"Loaded {aspect} aspect model from {model_path}")
                    
                except Exception as e:
                    logger.error(f"Error loading aspect model {model_path}: {str(e)}")
                    success = False
            
            self.trained = success and len(self.models) > 0
            return self.trained
            
        except Exception as e:
            logger.error(f"Error loading aspect models: {str(e)}")
            return False
    
    def analyze_aspect(self, text, aspect):
        """
        Analyze sentiment for a specific aspect.
        
        Args:
            text (str): Text to analyze
            aspect (str): Aspect to analyze
            
        Returns:
            tuple: (sentiment, confidence)
        """
        if not self.has_model(aspect):
            logger.warning(f"No trained model for aspect: {aspect}")
            # Fallback to rule-based for this aspect
            sentiment, score = aspect_based_sentiment(text, aspect)
            return sentiment, min(abs(score), 1.0)
        
        model = self.models[aspect]
        
        # Preprocess the text with advanced methods
        processed_text = preprocess_text(text, advanced=True)
        
        # Custom feature extraction for this aspect
        # Identify aspect-specific keywords in the text
        aspect_keywords = ASPECT_KEYWORDS.get(aspect, [])
        text_lower = processed_text.lower()
        
        # Extract context around aspect mentions for focused analysis
        aspect_context = ""
        words = text_lower.split()
        
        # Look for aspect keywords in text
        for i, word in enumerate(words):
            if any(keyword in word for keyword in aspect_keywords):
                # Extract a window around the keyword
                start = max(0, i - 5)
                end = min(len(words), i + 6)
                context_words = words[start:end]
                aspect_context += " ".join(context_words) + " "
        
        # If we found aspect context, use it; otherwise use the full text
        if aspect_context.strip():
            analysis_text = aspect_context
        else:
            analysis_text = processed_text
        
        # Make prediction
        try:
            prediction = model.pipeline.predict([analysis_text])[0]
            
            # Get confidence
            proba = model.pipeline.predict_proba([analysis_text])[0]
            class_indices = model.pipeline.classes_
            sentiment_index = list(class_indices).index(prediction)
            confidence = proba[sentiment_index]
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error predicting {aspect} sentiment: {str(e)}")
            # Fallback to rule-based
            sentiment, score = aspect_based_sentiment(text, aspect)
            return sentiment, min(abs(score), 1.0)
