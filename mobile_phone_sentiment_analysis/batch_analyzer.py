"""
Script for batch analyzing multiple YouTube videos.
"""

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from mobile_phone_sentiment_analysis.data.youtube_api import collect_comments_from_video
from mobile_phone_sentiment_analysis.models.sentiment_analyzer import SentimentAnalyzer
from mobile_phone_sentiment_analysis.utils.reporting import generate_full_report
from mobile_phone_sentiment_analysis.config.config import RESULTS_DIR

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def ensure_dirs():
    """Ensure required directories exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs('collected_comments', exist_ok=True)
    os.makedirs('batch_results', exist_ok=True)

def load_urls_from_file(file_path):
    """
    Load YouTube URLs from a file.
    
    Args:
        file_path (str): Path to the file containing URLs
        
    Returns:
        list: List of YouTube URLs
    """
    try:
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        logger.info(f"Loaded {len(urls)} URLs from {file_path}")
        return urls
    except Exception as e:
        logger.error(f"Error loading URLs from {file_path}: {str(e)}")
        return []

def analyze_video(video_url, analyzer, max_comments=100):
    """
    Collect and analyze comments from a YouTube video.
    
    Args:
        video_url (str): YouTube video URL
        analyzer (SentimentAnalyzer): Sentiment analyzer to use
        max_comments (int): Maximum number of comments to collect
        
    Returns:
        tuple: (analysis_results, video_details, report_paths)
    """
    try:
        logger.info(f"Analyzing video: {video_url}")
        
        # Collect comments from the video
        comments, video_details, csv_path = collect_comments_from_video(
            video_url, max_comments=max_comments, output_dir='collected_comments'
        )
        
        if not comments:
            logger.warning(f"No comments collected for video: {video_url}")
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
        
        return analysis_results, video_details, report_paths
    
    except Exception as e:
        logger.error(f"Error analyzing video {video_url}: {str(e)}")
        return None, None, None

def generate_batch_summary(results):
    """
    Generate a summary of the batch analysis.
    
    Args:
        results (dict): Dictionary mapping URLs to analysis results
        
    Returns:
        str: Path to the saved summary file
    """
    try:
        # Create a DataFrame for the summary
        summary_data = []
        
        for url, (analysis_results, video_details, report_paths) in results.items():
            if not video_details:
                continue
            
            # Count sentiments
            sentiments = [r.get('overall_sentiment', 'neutral') for r in analysis_results]
            positive_count = sentiments.count('positive')
            negative_count = sentiments.count('negative')
            neutral_count = sentiments.count('neutral')
            total_count = len(sentiments)
            
            # Calculate percentages
            positive_pct = (positive_count / total_count) * 100 if total_count > 0 else 0
            negative_pct = (negative_count / total_count) * 100 if total_count > 0 else 0
            neutral_pct = (neutral_count / total_count) * 100 if total_count > 0 else 0
            
            # Count aspects
            aspect_counts = {aspect: 0 for aspect in ['camera', 'battery', 'performance', 'display', 'design', 'price']}
            for result in analysis_results:
                for aspect in result.get('aspects', {}):
                    aspect_counts[aspect] += 1
            
            # Calculate most mentioned aspects
            sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)
            top_aspects = [f"{aspect} ({count})" for aspect, count in sorted_aspects[:3] if count > 0]
            top_aspects_str = ", ".join(top_aspects) if top_aspects else "None"
            
            # Add to summary data
            summary_data.append({
                'video_title': video_details['title'],
                'channel': video_details['channel_title'],
                'comments_analyzed': total_count,
                'positive_pct': positive_pct,
                'negative_pct': negative_pct,
                'neutral_pct': neutral_pct,
                'top_aspects': top_aspects_str,
                'video_url': url,
                'report_dir': RESULTS_DIR
            })
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join('batch_results', f'batch_summary_{timestamp}.csv')
        summary_df.to_csv(file_path, index=False)
        
        # Also save a text report
        text_report_path = os.path.join('batch_results', f'batch_summary_{timestamp}.txt')
        
        with open(text_report_path, 'w') as f:
            f.write("MOBILE PHONE SENTIMENT ANALYSIS - BATCH SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Videos Analyzed: {len(summary_data)}\n\n")
            
            for i, video in enumerate(summary_data):
                f.write(f"Video {i+1}: {video['video_title']}\n")
                f.write(f"Channel: {video['channel']}\n")
                f.write(f"URL: {video['video_url']}\n")
                f.write(f"Comments Analyzed: {video['comments_analyzed']}\n")
                f.write(f"Sentiment: Positive {video['positive_pct']:.1f}%, Negative {video['negative_pct']:.1f}%, Neutral {video['neutral_pct']:.1f}%\n")
                f.write(f"Top Mentioned Aspects: {video['top_aspects']}\n")
                f.write("\n" + "-" * 40 + "\n\n")
        
        logger.info(f"Saved batch summary to {file_path} and {text_report_path}")
        return text_report_path
    
    except Exception as e:
        logger.error(f"Error generating batch summary: {str(e)}")
        return None

def batch_analyze_videos(url_file, max_comments=100):
    """
    Analyze multiple YouTube videos from a file.
    
    Args:
        url_file (str): Path to file containing YouTube URLs
        max_comments (int): Maximum number of comments to collect per video
        
    Returns:
        dict: Analysis results for each video
    """
    # Ensure directories exist
    ensure_dirs()
    
    # Load URLs
    urls = load_urls_from_file(url_file)
    
    if not urls:
        logger.error("No URLs to analyze")
        return {}
    
    # Load sentiment analyzer
    logger.info("Loading sentiment analyzer...")
    analyzer = SentimentAnalyzer()
    
    try:
        analyzer.load_models()
        logger.info("Loaded existing sentiment models")
    except Exception as e:
        logger.warning(f"Could not load models, using rule-based analysis only: {str(e)}")
    
    # Analyze each video
    results = {}
    
    for i, url in enumerate(urls):
        logger.info(f"Processing video {i+1}/{len(urls)}: {url}")
        
        # Add delay to avoid API rate limits
        if i > 0:
            time.sleep(2)
        
        # Analyze video
        analysis_results, video_details, report_paths = analyze_video(
            url, analyzer, max_comments=max_comments
        )
        
        # Store results
        results[url] = (analysis_results, video_details, report_paths)
    
    # Generate batch summary
    summary_path = generate_batch_summary(results)
    
    logger.info(f"Batch analysis completed. Summary saved to {summary_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch analyze multiple YouTube videos')
    parser.add_argument('--file', required=True, help='File containing YouTube URLs (one per line)')
    parser.add_argument('--max', type=int, default=100, help='Maximum number of comments per video')
    
    args = parser.parse_args()
    
    logger.info(f"Starting batch analysis of videos from {args.file}")
    results = batch_analyze_videos(args.file, args.max)
    
    if results:
        logger.info(f"Completed analysis of {len(results)} videos")
    else:
        logger.error("Batch analysis failed") 