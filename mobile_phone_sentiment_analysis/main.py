"""
Main script for the mobile phone sentiment analysis system.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from mobile_phone_sentiment_analysis.data.data_loader import get_processed_data
from mobile_phone_sentiment_analysis.data.youtube_api import collect_comments_from_video
from mobile_phone_sentiment_analysis.models.sentiment_analyzer import SentimentAnalyzer
from mobile_phone_sentiment_analysis.utils.reporting import (
    print_report_summary, generate_full_report
)
from mobile_phone_sentiment_analysis.config.config import (
    RESULTS_DIR, MODEL_SAVE_DIR
)
from mobile_phone_sentiment_analysis.batch_analyzer import batch_analyze_videos

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mobile_phone_sentiment_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def ensure_dirs():
    """Ensure required directories exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def train_models(force_retrain=False):
    """
    Train sentiment analysis models.
    
    Args:
        force_retrain (bool): Force retraining even if models exist
        
    Returns:
        SentimentAnalyzer: Trained sentiment analyzer
    """
    # Create sentiment analyzer
    from mobile_phone_sentiment_analysis.models.sentiment_analyzer import SentimentAnalyzer
    
    # Create and load sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Check if models exist and no force retraining
    if not force_retrain and analyzer.load_models():
        logger.info("Loaded existing sentiment models")
        return analyzer
    
    # Get preprocessed data
    data = get_processed_data()
    
    # The function returns a tuple of (cleaned_df, train_test_splits, aspect_datasets)
    if len(data) != 3:
        logger.error(f"Unexpected data format returned from get_processed_data(): {len(data)} items instead of 3")
        return None
        
    cleaned_df, train_test_splits, aspect_datasets = data
    
    if train_test_splits is None or aspect_datasets is None:
        logger.error("No data available for training")
        return None
    
    # Train overall sentiment model
    logger.info("Training overall sentiment model...")
    analyzer.train_overall_model(
        train_test_splits['X_train'],
        train_test_splits['y_train']
    )
    
    # Evaluate overall model
    metrics = analyzer.evaluate_overall_model(
        train_test_splits['X_test'],
        train_test_splits['y_test']
    )
    
    logger.info(f"Overall sentiment model accuracy: {metrics['accuracy']:.4f}")
    
    # Train aspect-specific models
    logger.info("Training aspect-specific models...")
    aspect_metrics = analyzer.train_aspect_models(aspect_datasets)
    
    for aspect, metrics in aspect_metrics.items():
        # Check if metrics is a dict and has accuracy key (not an error)
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            logger.info(f"{aspect.capitalize()} model accuracy: {metrics['accuracy']:.4f}")
        elif isinstance(metrics, dict) and 'error' in metrics:
            logger.error(f"Error training {aspect} model: {metrics['error']}")
    
    # Save all models
    logger.info("Saving trained models...")
    analyzer.save_models()
    
    return analyzer

def analyze_video(video_url, analyzer=None, max_comments=100):
    """
    Collect and analyze comments from a YouTube video.
    
    Args:
        video_url (str): YouTube video URL
        analyzer (SentimentAnalyzer): Trained sentiment analyzer
        max_comments (int): Maximum number of comments to collect
        
    Returns:
        tuple: (analysis_results, video_details, report_paths)
    """
    # Ensure we have an analyzer
    if analyzer is None:
        analyzer = train_models()
        
    if analyzer is None:
        logger.error("No analyzer available for sentiment analysis")
        return None, None, None
    
    # Collect comments from the video
    logger.info(f"Collecting comments from YouTube video: {video_url}")
    comments, video_details, csv_path = collect_comments_from_video(
        video_url, max_comments=max_comments, output_dir='collected_comments'
    )
    
    if not comments:
        logger.error("No comments collected from video")
        return None, None, None
    
    # Analyze the comments
    logger.info(f"Analyzing {len(comments)} comments...")
    analysis_results = []
    
    for comment in comments:
        result = analyzer.analyze_sentiment(comment['comment_text'])
        
        # Add comment metadata
        result['comment_id'] = comment['comment_id']
        result['author'] = comment['author']
        result['like_count'] = comment['like_count']
        result['published_at'] = comment['published_at']
        
        analysis_results.append(result)
    
    # Generate report
    logger.info("Generating analysis report...")
    report_paths = generate_full_report(analysis_results, video_details['title'])
    
    # Print summary
    print_report_summary(analysis_results, video_details['title'])
    
    return analysis_results, video_details, report_paths

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Mobile Phone Sentiment Analysis System')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train models command
    train_parser = subparsers.add_parser('train', help='Train sentiment analysis models')
    train_parser.add_argument('--force', action='store_true', help='Force retraining of models')
    
    # Analyze video command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze YouTube video comments')
    analyze_parser.add_argument('--url', required=True, help='YouTube video URL')
    analyze_parser.add_argument('--max', type=int, default=100, help='Maximum number of comments to analyze')
    
    # Batch analyze command
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple YouTube videos from a file')
    batch_parser.add_argument('--file', required=True, help='File containing YouTube URLs (one per line)')
    batch_parser.add_argument('--max', type=int, default=100, help='Maximum number of comments per video')
    batch_parser.add_argument('--advanced', action='store_true', help='Use advanced batch processing with detailed reports')
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Ensure required directories exist
    ensure_dirs()
    
    # Parse command line arguments
    args = parse_arguments()
    
    if args.command == 'train':
        # Train models
        logger.info("Starting model training process...")
        analyzer = train_models(force_retrain=args.force)
        if analyzer:
            logger.info("Model training completed successfully")
        else:
            logger.error("Model training failed")
    
    elif args.command == 'analyze':
        # Analyze a single video
        logger.info(f"Analyzing YouTube video: {args.url}")
        analyzer = train_models()
        analysis_results, video_details, report_paths = analyze_video(
            args.url, analyzer, max_comments=args.max
        )
        
        if report_paths:
            logger.info(f"Analysis completed. Reports saved to: {RESULTS_DIR}")
        else:
            logger.error("Analysis failed")
    
    elif args.command == 'batch':
        # Analyze multiple videos from a file
        logger.info(f"Batch analyzing YouTube videos from: {args.file}")
        
        try:
            if args.advanced:
                # Use the advanced batch analyzer
                logger.info("Using advanced batch processing mode")
                results = batch_analyze_videos(args.file, max_comments=args.max)
                
                if results:
                    logger.info(f"Advanced batch analysis completed. Processed {len(results)} videos")
                else:
                    logger.error("Advanced batch analysis failed")
            else:
                # Use the simple batch processing approach
                # Load URLs from file
                with open(args.file, 'r') as f:
                    urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                
                logger.info(f"Found {len(urls)} URLs to analyze")
                
                # Train models once for all videos
                analyzer = train_models()
                
                # Analyze each video
                for i, url in enumerate(urls):
                    logger.info(f"Analyzing video {i+1}/{len(urls)}: {url}")
                    analyze_video(url, analyzer, max_comments=args.max)
                
                logger.info("Batch analysis completed")
        
        except Exception as e:
            logger.error(f"Batch analysis failed: {str(e)}")
    
    else:
        # No command or invalid command
        logger.error("No valid command provided. Use --help for usage information.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
