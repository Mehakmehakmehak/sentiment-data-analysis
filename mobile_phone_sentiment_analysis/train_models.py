"""
Script for training and evaluating sentiment analysis models.
"""

import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from mobile_phone_sentiment_analysis.data.data_loader import get_processed_data
from mobile_phone_sentiment_analysis.models.sentiment_analyzer import SentimentAnalyzer
from mobile_phone_sentiment_analysis.config.config import MODEL_SAVE_DIR, RESULTS_DIR

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_training.log')
    ]
)
logger = logging.getLogger(__name__)

def ensure_dirs():
    """Ensure required directories exist."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_evaluate_models(force_retrain=False):
    """
    Train and evaluate all sentiment analysis models.
    
    Args:
        force_retrain (bool): Whether to force retraining even if models exist
        
    Returns:
        dict: Evaluation metrics for all models
    """
    # Ensure directories exist
    ensure_dirs()
    
    # Create sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Check if models already exist
    if not force_retrain:
        try:
            logger.info("Checking for existing models...")
            if analyzer.load_models():
                logger.info("Loaded existing models successfully. Use --force to retrain.")
                return evaluate_existing_models(analyzer)
        except Exception as e:
            logger.warning(f"Could not load existing models: {str(e)}")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing annotated data...")
    cleaned_df, train_test_splits, aspect_datasets = get_processed_data()
    
    if cleaned_df is None:
        logger.error("No data available for training")
        return None
    
    # Train overall sentiment model
    logger.info("Training overall sentiment model...")
    analyzer.train_overall_model(
        train_test_splits['X_train'],
        train_test_splits['y_train']
    )
    
    # Evaluate overall model
    overall_metrics = analyzer.evaluate_overall_model(
        train_test_splits['X_test'],
        train_test_splits['y_test']
    )
    
    logger.info(f"Overall sentiment model accuracy: {overall_metrics['accuracy']:.4f}")
    
    # Plot confusion matrix for overall model
    plot_confusion_matrix(
        analyzer.overall_model, 
        [preprocess_text(text) for text in train_test_splits['X_test']], 
        train_test_splits['y_test'],
        'overall'
    )
    
    # Train aspect-specific models
    logger.info("Training aspect-specific models...")
    aspect_metrics = analyzer.train_aspect_models(aspect_datasets)
    
    for aspect, metrics in aspect_metrics.items():
        if metrics:
            logger.info(f"{aspect.capitalize()} model accuracy: {metrics['accuracy']:.4f}")
    
    # Save all models
    logger.info("Saving trained models...")
    analyzer.save_models()
    
    # Return all metrics
    all_metrics = {
        'overall': overall_metrics,
        'aspects': aspect_metrics
    }
    
    return all_metrics

def evaluate_existing_models(analyzer):
    """
    Evaluate existing models on the test data.
    
    Args:
        analyzer (SentimentAnalyzer): Loaded sentiment analyzer
        
    Returns:
        dict: Evaluation metrics for all models
    """
    # Load and preprocess data
    logger.info("Loading data for evaluation...")
    cleaned_df, train_test_splits, aspect_datasets = get_processed_data()
    
    if cleaned_df is None:
        logger.error("No data available for evaluation")
        return None
    
    # Evaluate overall model
    overall_metrics = analyzer.evaluate_overall_model(
        train_test_splits['X_test'],
        train_test_splits['y_test']
    )
    
    logger.info(f"Overall sentiment model accuracy: {overall_metrics['accuracy']:.4f}")
    
    # Plot confusion matrix for overall model
    plot_confusion_matrix(
        analyzer.overall_model, 
        [preprocess_text(text) for text in train_test_splits['X_test']], 
        train_test_splits['y_test'],
        'overall'
    )
    
    # Evaluate aspect models
    aspect_metrics = {}
    
    for aspect, df in aspect_datasets.items():
        # Create train/test split for this aspect
        from sklearn.model_selection import train_test_split
        from mobile_phone_sentiment_analysis.config.config import TEST_SIZE, RANDOM_STATE
        
        X = df['comment_text']
        y = df[f'{aspect}_sentiment']
        
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        if aspect in analyzer.aspect_manager.models:
            # Evaluate this aspect model
            model = analyzer.aspect_manager.models[aspect]
            metrics = model.evaluate(X_test, y_test)
            aspect_metrics[aspect] = metrics
            
            logger.info(f"{aspect.capitalize()} model accuracy: {metrics['accuracy']:.4f}")
    
    # Return all metrics
    all_metrics = {
        'overall': overall_metrics,
        'aspects': aspect_metrics
    }
    
    return all_metrics

def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Plot and save a confusion matrix for model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model (for file naming)
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get classes
        classes = model.classes_
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        
        # Plot
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix - {model_name.capitalize()} Sentiment Model')
        
        # Save
        plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png'))
        plt.close()
        
        logger.info(f"Saved confusion matrix for {model_name} model")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")

def preprocess_text(text):
    """Preprocess text for model evaluation."""
    from mobile_phone_sentiment_analysis.utils.preprocessing import preprocess_text as pp
    return pp(text)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate sentiment analysis models')
    parser.add_argument('--force', action='store_true', help='Force retraining even if models exist')
    args = parser.parse_args()
    
    logger.info("Starting model training and evaluation...")
    metrics = train_and_evaluate_models(force_retrain=args.force)
    
    if metrics:
        logger.info("Training and evaluation completed successfully")
    else:
        logger.error("Training and evaluation failed") 