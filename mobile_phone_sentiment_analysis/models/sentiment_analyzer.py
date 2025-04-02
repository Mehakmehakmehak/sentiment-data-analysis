"""
Main sentiment analyzer for the mobile phone sentiment analysis system.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import logging
import nltk
import random
from sklearn.base import BaseEstimator, TransformerMixin

# Import configuration and utilities
sys.path.append(".")
from mobile_phone_sentiment_analysis.config.config import MODEL_SAVE_DIR, MAX_FEATURES, N_ESTIMATORS, RANDOM_STATE, C_VALUE, MAX_ITER, CLASS_WEIGHT
from mobile_phone_sentiment_analysis.config.config import POSITIVE_WORDS, NEGATIVE_WORDS, NEUTRAL_WORDS, INTENSIFIERS, NEGATORS
from mobile_phone_sentiment_analysis.config.config import DATA_AUGMENTATION, ENSEMBLE_WEIGHTS, CROSS_VALIDATION_FOLDS
from mobile_phone_sentiment_analysis.utils.preprocessing import preprocess_text, preprocess_with_spacy, nlp
from mobile_phone_sentiment_analysis.utils.preprocessing import synonym_replacement, random_swap
from mobile_phone_sentiment_analysis.utils.feature_extraction import (
    extract_aspects, rule_based_sentiment, aspect_based_sentiment
)
from mobile_phone_sentiment_analysis.models.aspect_models import AspectModelManager

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def augment_data(texts, labels, multiplier=1):
    """Augment training data using NLP techniques."""
    if not DATA_AUGMENTATION.get('enabled', False):
        return texts, labels
    
    logger.info(f"Augmenting training data with multiplier {multiplier}...")
    
    # Convert pandas Series to list if necessary
    texts_list = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
    labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
    
    augmented_texts = texts_list.copy()
    augmented_labels = labels_list.copy()
    
    # Only augment texts that are longer than 10 words
    eligible_indices = [i for i, text in enumerate(texts_list) if len(str(text).split()) > 10]
    
    if not eligible_indices:
        logger.warning("No eligible texts found for augmentation")
        return texts, labels
    
    # Randomly select indices for augmentation based on multiplier
    num_to_augment = int(min(len(eligible_indices), len(texts_list) * multiplier))
    if num_to_augment <= 0:
        return texts, labels
        
    indices_to_augment = random.sample(eligible_indices, num_to_augment)
    
    for idx in indices_to_augment:
        text = texts_list[idx]
        label = labels_list[idx]
        
        # Apply synonym replacement
        if DATA_AUGMENTATION.get('synonym_replacement', True):
            try:
                aug_text = synonym_replacement(str(text), n=2)
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
            except Exception as e:
                logger.warning(f"Error in synonym replacement: {str(e)}")
        
        # Apply random swap
        if DATA_AUGMENTATION.get('random_swap', True):
            try:
                aug_text = random_swap(str(text), n=2)
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
            except Exception as e:
                logger.warning(f"Error in random swap: {str(e)}")
    
    logger.info(f"Data augmentation complete. Original size: {len(texts_list)}, Augmented size: {len(augmented_texts)}")
    return augmented_texts, augmented_labels

# Custom feature extractors for sentiment analysis
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """Extract text length features"""
    def fit(self, x, y=None):
        return self
    
    def transform(self, texts):
        return np.array([[
            len(text),  # Total length
            len(text.split()),  # Word count
            sum(c == '!' for c in text),  # Exclamation count
            sum(c == '?' for c in text),  # Question mark count
            len(text) / max(len(text.split()), 1),  # Average word length
            sum(c.isupper() for c in text) / max(len(text), 1),  # Proportion of uppercase chars
            text.count('.') + text.count('...'),  # Period count
        ] for text in texts])

class SentimentLexiconFeatures(BaseEstimator, TransformerMixin):
    """Extract features based on sentiment lexicons"""
    def fit(self, x, y=None):
        return self
    
    def transform(self, texts):
        features = []
        for text in texts:
            text = text.lower()
            words = text.split()
            
            # Count sentiment words
            pos_count = sum(1 for word in words if word in POSITIVE_WORDS)
            neg_count = sum(1 for word in words if word in NEGATIVE_WORDS)
            neutral_count = sum(1 for word in words if word in NEUTRAL_WORDS)
            
            # Count intensifiers and negators
            intensifier_count = sum(1 for word in words if word in INTENSIFIERS)
            negator_count = sum(1 for word in words if word in NEGATORS)
            
            # Check for negated positive and negative words
            negation_window = 3
            negated_pos = 0
            negated_neg = 0
            
            for i, word in enumerate(words):
                if word in NEGATORS:
                    # Check the next few words for sentiment words
                    window_end = min(i + negation_window + 1, len(words))
                    for j in range(i+1, window_end):
                        if words[j] in POSITIVE_WORDS:
                            negated_pos += 1
                        elif words[j] in NEGATIVE_WORDS:
                            negated_neg += 1
            
            # Calculate ratios if there are words
            word_count = len(words) if words else 1
            pos_ratio = pos_count / word_count
            neg_ratio = neg_count / word_count
            neutral_ratio = neutral_count / word_count
            
            # Sentiment score with negation handling
            effective_pos = pos_count - negated_pos  # Positive words not negated
            effective_neg = neg_count - negated_neg + negated_pos  # Negative words not negated + negated positives
            sentiment_score = (effective_pos - effective_neg) / word_count
            
            features.append([
                pos_count,
                neg_count,
                neutral_count,
                pos_count - neg_count,  # Raw sentiment difference
                sentiment_score,  # Negation-aware sentiment score
                pos_ratio,
                neg_ratio,
                neutral_ratio,
                intensifier_count / word_count,  # Intensifier ratio
                negator_count / word_count,  # Negator ratio
                negated_pos / max(pos_count, 1),  # Proportion of positives that are negated
                negated_neg / max(neg_count, 1),  # Proportion of negatives that are negated
                1 if pos_count > neg_count else (-1 if neg_count > pos_count else 0)  # Simple polarity
            ])
        
        return np.array(features)

class ContextualPatternFeatures(BaseEstimator, TransformerMixin):
    """Extract contextual patterns like "but", "however" that may signal sentiment shifts"""
    def fit(self, x, y=None):
        return self
    
    def transform(self, texts):
        # Phrases that may indicate sentiment shift
        contrast_phrases = ['but', 'however', 'although', 'though', 'nonetheless', 
                          'nevertheless', 'yet', 'still', 'while', 'despite', 'in spite of']
        
        features = []
        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()
            
            # Check for contrast phrases
            contrast_count = sum(1 for phrase in contrast_phrases if phrase in words)
            
            # Check for sentiment shifts around contrast phrases
            sentiment_shifts = 0
            for i, word in enumerate(words):
                if word in contrast_phrases:
                    # Analyze sentiment before and after the contrast phrase
                    pre_words = words[:i]
                    post_words = words[i+1:]
                    
                    pre_pos = sum(1 for w in pre_words if w in POSITIVE_WORDS)
                    pre_neg = sum(1 for w in pre_words if w in NEGATIVE_WORDS)
                    post_pos = sum(1 for w in post_words if w in POSITIVE_WORDS)
                    post_neg = sum(1 for w in post_words if w in NEGATIVE_WORDS)
                    
                    pre_sentiment = 1 if pre_pos > pre_neg else (-1 if pre_neg > pre_pos else 0)
                    post_sentiment = 1 if post_pos > post_neg else (-1 if post_neg > post_pos else 0)
                    
                    if pre_sentiment != post_sentiment and pre_sentiment != 0 and post_sentiment != 0:
                        sentiment_shifts += 1
            
            # Features for conditional phrases
            conditional_phrases = ['if', 'would', 'could', 'might', 'may']
            conditional_count = sum(1 for phrase in conditional_phrases if phrase in words)
            
            # Features for emphasis
            emphasis_patterns = ['!', 'really', 'very', 'absolutely', 'definitely', 'certainly']
            emphasis_count = sum(text_lower.count(pattern) for pattern in emphasis_patterns)
            
            features.append([
                contrast_count,
                sentiment_shifts,
                conditional_count,
                emphasis_count,
                contrast_count > 0,  # Binary feature for presence of contrast
                sentiment_shifts > 0,  # Binary feature for presence of sentiment shift
                conditional_count > 0,  # Binary feature for conditional language
            ])
        
        return np.array(features)

def ensure_model_dir():
    """Ensure the model save directory exists."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    logger.info(f"Ensuring model directory exists: {MODEL_SAVE_DIR}")

class SentimentAnalyzer:
    """
    Main sentiment analyzer class that combines ML and rule-based approaches.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.overall_model = None
        self.aspect_manager = AspectModelManager()
        self.trained = False
    
    def build_overall_model(self):
        """
        Build the overall sentiment model pipeline using an advanced text classification approach.
        """
        # Base SVM classifier
        base_svm = LinearSVC(
            C=C_VALUE,
            class_weight=CLASS_WEIGHT,
            dual=False,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
            loss='squared_hinge'
        )
        
        # SVM with calibration to get probability estimates
        calibrated_svm = CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')
        
        # Gradient Boosting classifier for ensemble
        gb_classifier = GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE
        )
        
        # Feature pipeline with multiple feature types
        feature_pipeline = FeatureUnion([
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
                binary=True,
                stop_words='english'
            )),
            ('text_stats', TextLengthExtractor()),
            ('lexicon_features', SentimentLexiconFeatures()),
            ('contextual_patterns', ContextualPatternFeatures())
        ])
        
        # Create voting ensemble for improved accuracy
        ensemble = VotingClassifier(
            estimators=[
                ('svm', calibrated_svm),
                ('gb', gb_classifier)
            ],
            voting='soft',
            weights=[0.7, 0.3]  # Weight SVM higher as it typically performs better for text
        )
        
        # Create pipeline with features and classifier
        self.overall_model = Pipeline([
            ('features', feature_pipeline),
            ('classifier', ensemble)
        ])
        
        logger.info("Built advanced ensemble sentiment model with SVM and Gradient Boosting")
    
    def train_overall_model(self, X_train, y_train):
        """
        Train the overall sentiment model.
        
        Args:
            X_train (array-like): Training text data
            y_train (array-like): Training sentiment labels
            
        Returns:
            self: The trained model instance
        """
        if self.overall_model is None:
            self.build_overall_model()
        
        # Augment training data
        X_train_aug, y_train_aug = augment_data(X_train, y_train, multiplier=0.3)
        
        # Preprocess the text data
        X_train_processed = [preprocess_text(text) for text in X_train_aug]
        
        # Cross-validation to evaluate model before final training
        if CROSS_VALIDATION_FOLDS > 1:
            logger.info(f"Performing {CROSS_VALIDATION_FOLDS}-fold cross-validation...")
            cv_scores = cross_val_score(
                self.overall_model, X_train_processed, y_train_aug, 
                cv=CROSS_VALIDATION_FOLDS, scoring='accuracy'
            )
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train the model on all training data
        logger.info(f"Training final overall sentiment model on {len(X_train_processed)} samples")
        self.overall_model.fit(X_train_processed, y_train_aug)
        self.trained = True
        
        return self
    
    def evaluate_overall_model(self, X_test, y_test):
        """
        Evaluate the overall sentiment model.
        
        Args:
            X_test (array-like): Test text data
            y_test (array-like): Test sentiment labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.trained or self.overall_model is None:
            logger.error("Cannot evaluate overall model - not trained yet")
            return None
        
        # Preprocess the text data
        X_test_processed = [preprocess_text(text) for text in X_test]
        
        # Make predictions
        y_pred = self.overall_model.predict(X_test_processed)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Evaluated overall sentiment model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Return metrics
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': report
        }
    
    def train_aspect_models(self, aspect_datasets):
        """
        Train aspect-specific sentiment models.
        
        Args:
            aspect_datasets (dict): Dictionary mapping aspects to DataFrames
            
        Returns:
            dict: Evaluation metrics for each aspect model
        """
        return self.aspect_manager.train_aspect_models(aspect_datasets)
    
    def save_models(self):
        """
        Save all trained models to disk.
        
        Returns:
            bool: True if all models were saved successfully
        """
        ensure_model_dir()
        
        if not self.trained or self.overall_model is None:
            logger.error("Cannot save overall model - not trained yet")
            return False
        
        # Save overall model
        overall_model_path = os.path.join(MODEL_SAVE_DIR, "overall_sentiment_model.joblib")
        joblib.dump(self.overall_model, overall_model_path)
        logger.info(f"Saved overall sentiment model to {overall_model_path}")
        
        return True
    
    def load_models(self):
        """
        Load all models from disk.
        
        Returns:
            bool: True if all models were loaded successfully
        """
        # Load overall model
        overall_model_path = os.path.join(MODEL_SAVE_DIR, "overall_sentiment_model.joblib")
        
        try:
            self.overall_model = joblib.load(overall_model_path)
            self.trained = True
            logger.info(f"Loaded overall sentiment model from {overall_model_path}")
        except Exception as e:
            logger.error(f"Error loading overall model: {str(e)}")
            return False
        
        # Load aspect models
        aspect_success = self.aspect_manager.load_all_models()
        
        return aspect_success and self.trained
    
    def predict_overall_sentiment(self, text):
        """
        Predict overall sentiment for a text using the ML model.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Predicted sentiment
            float: Confidence score
        """
        if not self.trained or self.overall_model is None:
            logger.error("Cannot predict with overall model - not trained yet")
            return "neutral", 0.0
        
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Make prediction
        prediction = self.overall_model.predict([processed_text])[0]
        
        # Get probability for the predicted class
        proba = self.overall_model.predict_proba([processed_text])[0]
        class_indices = self.overall_model.classes_
        sentiment_index = list(class_indices).index(prediction)
        confidence = proba[sentiment_index]
        
        return prediction, confidence
    
    def analyze_sentiment(self, text):
        """
        Hybrid sentiment analysis combining ML and rule-based approaches.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Sentiment analysis results
        """
        # Initialize results
        results = {
            'text': text,
            'overall_sentiment': None,
            'confidence': 0.0,
            'aspects': {}
        }
        
        # Get ML-based sentiment and confidence
        ml_sentiment, ml_confidence = self.predict_overall_sentiment(text)
        
        # Get rule-based sentiment and score
        rule_sentiment, rule_score = rule_based_sentiment(text)
        
        # Normalize rule score to 0-1 range (make sure it's not above 1)
        rule_confidence = min(abs(rule_score), 1.0)
        
        # Enhanced hybrid approach for overall sentiment with weighted ensemble
        # Use ensemble weights from config
        ml_weight = ENSEMBLE_WEIGHTS.get('svm', 0.7)
        rule_weight = ENSEMBLE_WEIGHTS.get('rule_based', 0.3)
        
        # Apply weighting logic
        if ml_confidence > 0.85:
            # Very high ML confidence, trust ML more
            final_sentiment = ml_sentiment
            final_confidence = ml_confidence
        elif rule_confidence > 0.75 and ml_confidence < 0.6:
            # Strong rule signal and weak ML signal
            final_sentiment = rule_sentiment
            final_confidence = rule_confidence
        else:
            # Use weighted voting with confidence-adjusted weights
            sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
            sentiments[ml_sentiment] += ml_confidence * ml_weight
            sentiments[rule_sentiment] += rule_confidence * rule_weight
            
            # Get the sentiment with highest weighted vote
            final_sentiment = max(sentiments, key=sentiments.get)
            final_confidence = sentiments[final_sentiment] / (ml_weight + rule_weight)
        
        results['overall_sentiment'] = final_sentiment
        results['confidence'] = final_confidence
        
        # Extract mentioned aspects
        aspects = extract_aspects(text, nlp)
        
        # Analyze sentiment for each mentioned aspect
        for aspect, mentioned in aspects.items():
            if mentioned:
                # Apply enhanced hybrid approach for aspect sentiment
                
                # Get sentiment for this aspect using ML model
                aspect_ml_sentiment = "neutral"  # Default
                aspect_ml_confidence = 0.0
                
                # Try ML model first
                if self.aspect_manager.has_model(aspect):
                    aspect_ml_sentiment, aspect_ml_confidence = self.aspect_manager.analyze_aspect(text, aspect)
                
                # Get rule-based aspect sentiment
                aspect_rule_sentiment, aspect_rule_score = aspect_based_sentiment(text, aspect)
                aspect_rule_confidence = min(abs(aspect_rule_score), 1.0)
                
                # Combine ML and rule-based predictions with enhanced logic
                if aspect_ml_confidence > 0.8:
                    # High ML confidence, trust ML more
                    final_aspect_sentiment = aspect_ml_sentiment
                    final_aspect_confidence = aspect_ml_confidence
                elif aspect_rule_confidence > 0.7 and aspect_ml_confidence < 0.6:
                    # Strong rule signal and weak ML signal
                    final_aspect_sentiment = aspect_rule_sentiment
                    final_aspect_confidence = aspect_rule_confidence
                else:
                    # Use weighted voting with adjusted weights
                    aspect_sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
                    aspect_sentiments[aspect_ml_sentiment] += aspect_ml_confidence * ml_weight
                    aspect_sentiments[aspect_rule_sentiment] += aspect_rule_confidence * rule_weight
                    
                    # Get the sentiment with highest weighted vote
                    final_aspect_sentiment = max(aspect_sentiments, key=aspect_sentiments.get)
                    final_aspect_confidence = aspect_sentiments[final_aspect_sentiment] / (ml_weight + rule_weight)
                
                # Convert any non-string sentiment values to strings
                if not isinstance(final_aspect_sentiment, str):
                    final_aspect_sentiment = str(final_aspect_sentiment)
                
                # Store the final sentiment and confidence
                results['aspects'][aspect] = {
                    'sentiment': final_aspect_sentiment,
                    'confidence': final_aspect_confidence
                }
        
        return results
    
    def analyze_batch(self, texts):
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: List of sentiment analysis results
        """
        return [self.analyze_sentiment(text) for text in texts]
