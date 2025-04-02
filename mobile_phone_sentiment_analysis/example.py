"""
Example script to demonstrate the mobile phone sentiment analysis system.
"""

import os
import sys
import logging
from pprint import pprint

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from mobile_phone_sentiment_analysis.models.sentiment_analyzer import SentimentAnalyzer
from mobile_phone_sentiment_analysis.utils.preprocessing import preprocess_text
from mobile_phone_sentiment_analysis.utils.feature_extraction import extract_aspects, nlp
from mobile_phone_sentiment_analysis.data.youtube_api import collect_comments_from_video
from mobile_phone_sentiment_analysis.utils.reporting import print_report_summary

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_sample_comments():
    """
    Analyze a set of sample comments to demonstrate the system.
    """
    # Sample comments about mobile phones
    sample_comments = [
        "The camera on this phone is absolutely amazing! I've taken some incredible photos, especially in low light.",
        "Battery life is terrible, I have to charge it twice a day even with minimal use.",
        "The screen is gorgeous with vibrant colors, but the high refresh rate drains the battery quickly.",
        "Overall a decent phone, but way overpriced for what you get. Wait for a sale.",
        "Performance is smooth for everyday tasks, but it struggles with heavy gaming and gets really hot."
    ]
    
    logger.info("Initializing sentiment analyzer...")
    
    # Initialize the analyzer with rule-based approach only (no ML models)
    analyzer = SentimentAnalyzer()
    
    # Analyze each comment
    logger.info("Analyzing sample comments...")
    
    results = []
    for i, comment in enumerate(sample_comments):
        print(f"\nSample Comment #{i+1}:")
        print(f"\"{comment}\"")
        
        # Extract aspects mentioned in the comment
        aspects = extract_aspects(comment, nlp)
        mentioned_aspects = [aspect for aspect, mentioned in aspects.items() if mentioned]
        
        print(f"Aspects mentioned: {', '.join(mentioned_aspects) if mentioned_aspects else 'None'}")
        
        # Analyze sentiment
        result = analyzer.analyze_sentiment(comment)
        
        # Print sentiment results
        print(f"Overall sentiment (rule-based): {result['rule_based_sentiment']} (score: {result['rule_based_score']:.2f})")
        
        # Print aspect sentiments
        if result['aspects']:
            print("Aspect sentiments:")
            for aspect, details in result['aspects'].items():
                if isinstance(details, dict) and 'rule_sentiment' in details:
                    print(f"  - {aspect}: {details['rule_sentiment']} (score: {details['rule_score']:.2f})")
                
        results.append(result)
    
    return results

def analyze_youtube_video(video_url, max_comments=5):
    """
    Demonstrate analyzing comments from a YouTube video.
    
    Args:
        video_url (str): URL of a YouTube video
        max_comments (int): Maximum number of comments to analyze
    """
    logger.info(f"Collecting and analyzing comments from: {video_url}")
    
    # Collect comments from the video
    comments, video_details, _ = collect_comments_from_video(
        video_url, max_comments=max_comments
    )
    
    if not comments:
        logger.error("Failed to collect comments")
        return
    
    print(f"\nAnalyzing {len(comments)} comments from video: \"{video_details['title']}\"")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze comments
    results = []
    for comment in comments:
        result = analyzer.analyze_sentiment(comment['comment_text'])
        result['author'] = comment['author']
        result['like_count'] = comment['like_count']
        results.append(result)
    
    # Print summary
    print_report_summary(results, video_details['title'])
    
    return results

if __name__ == "__main__":
    # Example 1: Analyze sample comments
    print("=" * 80)
    print("EXAMPLE 1: ANALYZING SAMPLE COMMENTS")
    print("=" * 80)
    analyze_sample_comments()
    
    # Example 2: Analyze YouTube video
    # Uncomment and replace with a real video URL to run this example
    print("\n" + "=" * 80)
    print("EXAMPLE 2: ANALYZING YOUTUBE VIDEO COMMENTS")
    print("=" * 80)
    print("Note: This example requires a valid YouTube API key in the config.")
    print("      Uncomment the code in example.py to run this example with a real video URL.")
    
    # Sample URL for a phone review video (replace with a real URL)
    # video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    # analyze_youtube_video(video_url, max_comments=5) 