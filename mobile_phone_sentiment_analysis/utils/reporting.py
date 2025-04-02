"""
Reporting utilities for the mobile phone sentiment analysis system.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from wordcloud import WordCloud
import logging
import numpy as np
from collections import Counter

# Import configuration
sys.path.append(".")
from mobile_phone_sentiment_analysis.config.config import RESULTS_DIR

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default results directory - this will be overridden by Flask app
RESULTS_DIR = RESULTS_DIR 

def ensure_results_dir(custom_dir=None):
    """
    Ensure that the results directory exists.
    
    Args:
        custom_dir (str): Optional custom directory path
    """
    global RESULTS_DIR
    
    if custom_dir:
        RESULTS_DIR = custom_dir
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger.info(f"Ensuring results directory exists: {RESULTS_DIR}")

def generate_sentiment_summary(results, video_title):
    """
    Generate a text summary of sentiment analysis results.
    
    Args:
        results (list): List of analysis results
        video_title (str): Title of the video
        
    Returns:
        str: Formatted summary text
    """
    if not results:
        return "No results to summarize."
    
    # Count overall sentiments
    sentiments = [r.get('overall_sentiment', 'neutral') for r in results]
    sentiment_counts = Counter(sentiments)
    total_comments = len(results)
    
    # Count aspect sentiments
    aspect_sentiments = {
        'camera': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
        'battery': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
        'performance': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
        'display': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
        'design': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
        'price': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
    }
    
    for r in results:
        for aspect, sentiment in r.get('aspects', {}).items():
            if aspect in aspect_sentiments:
                # Handle case where sentiment might be a dictionary or non-string type
                if isinstance(sentiment, dict):
                    # Skip dictionaries as they can't be used as keys
                    continue
                elif not isinstance(sentiment, str):
                    # Convert to string if it's not already
                    sentiment = str(sentiment)
                
                # Only count if sentiment is one of the expected values
                if sentiment in ['positive', 'negative', 'neutral']:
                    aspect_sentiments[aspect][sentiment] += 1
                    aspect_sentiments[aspect]['total'] += 1
    
    # Generate report
    summary = []
    summary.append(f"Sentiment Analysis Report for: {video_title}")
    summary.append("=" * 50)
    summary.append(f"Total comments analyzed: {total_comments}")
    
    summary.append("\nOverall Sentiment:")
    for sentiment in ['positive', 'negative', 'neutral']:
        count = sentiment_counts.get(sentiment, 0)
        percentage = (count / total_comments) * 100 if total_comments > 0 else 0
        summary.append(f"{sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    summary.append("\nAspect-Based Sentiment:")
    for aspect, counts in aspect_sentiments.items():
        total = counts['total']
        if total > 0:
            summary.append(f"\n{aspect.capitalize()} (mentioned in {total} comments, {(total/total_comments)*100:.1f}% of total):")
            for sentiment in ['positive', 'negative', 'neutral']:
                count = counts.get(sentiment, 0)
                percentage = (count / total) * 100 if total > 0 else 0
                summary.append(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    return '\n'.join(summary)

def save_sentiment_summary(summary, video_title):
    """
    Save the sentiment summary to a text file.
    
    Args:
        summary (str): Sentiment summary text
        video_title (str): Title of the video
        
    Returns:
        str: Path to the saved file
    """
    ensure_results_dir()
    
    # Clean up the video title for use as a filename
    clean_title = ''.join(c if c.isalnum() else '_' for c in video_title)
    clean_title = clean_title[:50]  # Truncate if too long
    
    file_path = os.path.join(RESULTS_DIR, f"{clean_title}_sentiment_summary.txt")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    logger.info(f"Saved sentiment summary to {file_path}")
    return file_path

def generate_sentiment_pie_chart(results, video_title):
    """
    Generate a pie chart of overall sentiment distribution.
    
    Args:
        results (list): List of analysis results
        video_title (str): Title of the video
        
    Returns:
        str: Path to the saved chart
    """
    ensure_results_dir()
    
    # Count overall sentiments
    sentiments = [r.get('overall_sentiment', 'neutral') for r in results]
    sentiment_counts = Counter(sentiments)
    
    # Get counts for each sentiment
    positive_count = sentiment_counts.get('positive', 0)
    negative_count = sentiment_counts.get('negative', 0)
    neutral_count = sentiment_counts.get('neutral', 0)
    
    # Generate pie chart
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive_count, negative_count, neutral_count]
    colors = ['#4CAF50', '#F44336', '#9E9E9E']
    explode = (0.1, 0.1, 0.1)  # explode all slices
    
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    plt.title(f'Overall Sentiment Distribution for\n"{video_title}"', fontsize=14)
    
    # Clean up the video title for use as a filename
    clean_title = ''.join(c if c.isalnum() else '_' for c in video_title)
    clean_title = clean_title[:50]  # Truncate if too long
    
    file_path = os.path.join(RESULTS_DIR, f"{clean_title}_sentiment_pie.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved sentiment pie chart to {file_path}")
    return file_path

def generate_aspect_comparison_chart(results, video_title):
    """
    Generate a bar chart comparing sentiment across different aspects.
    
    Args:
        results (list): List of analysis results
        video_title (str): Title of the video
        
    Returns:
        str: Path to the saved chart
    """
    ensure_results_dir()
    
    # Count aspect sentiments
    aspect_sentiments = {
        'camera': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
        'battery': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
        'performance': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
        'display': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
        'design': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
        'price': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
    }
    
    for r in results:
        for aspect, sentiment in r.get('aspects', {}).items():
            if aspect in aspect_sentiments:
                # Handle case where sentiment might be a dictionary or non-string type
                if isinstance(sentiment, dict):
                    # Skip dictionaries as they can't be used as keys
                    continue
                elif not isinstance(sentiment, str):
                    # Convert to string if it's not already
                    sentiment = str(sentiment)
                
                # Only count if sentiment is one of the expected values
                if sentiment in ['positive', 'negative', 'neutral']:
                    aspect_sentiments[aspect][sentiment] += 1
                    aspect_sentiments[aspect]['total'] += 1
    
    # Prepare data for plotting
    aspects = []
    positive_counts = []
    negative_counts = []
    neutral_counts = []
    
    for aspect, counts in aspect_sentiments.items():
        if counts['total'] > 0:  # Only include aspects that were mentioned
            aspects.append(aspect.capitalize())
            
            # Convert to percentages
            total = counts['total']
            positive_counts.append((counts['positive'] / total) * 100 if total > 0 else 0)
            negative_counts.append((counts['negative'] / total) * 100 if total > 0 else 0)
            neutral_counts.append((counts['neutral'] / total) * 100 if total > 0 else 0)
    
    if not aspects:
        logger.warning("No aspects found for comparison chart")
        return None
    
    # Create bar chart
    x = np.arange(len(aspects))
    width = 0.25
    
    fig, ax = plt.figure(figsize=(12, 8)), plt.axes()
    rects1 = ax.bar(x - width, positive_counts, width, label='Positive', color='#4CAF50')
    rects2 = ax.bar(x, negative_counts, width, label='Negative', color='#F44336')
    rects3 = ax.bar(x + width, neutral_counts, width, label='Neutral', color='#9E9E9E')
    
    # Add labels, title and legend
    ax.set_ylabel('Percentage of Comments', fontsize=12)
    ax.set_title(f'Aspect-Based Sentiment Analysis for\n"{video_title}"', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    plt.tight_layout()
    
    # Clean up the video title for use as a filename
    clean_title = ''.join(c if c.isalnum() else '_' for c in video_title)
    clean_title = clean_title[:50]  # Truncate if too long
    
    file_path = os.path.join(RESULTS_DIR, f"{clean_title}_aspect_comparison.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved aspect comparison chart to {file_path}")
    return file_path

def generate_word_clouds(results, video_title):
    """
    Generate word clouds for positive and negative comments.
    
    Args:
        results (list): List of analysis results
        video_title (str): Title of the video
        
    Returns:
        tuple: (positive_cloud_path, negative_cloud_path)
    """
    ensure_results_dir()
    
    def create_word_cloud(texts, sentiment):
        """Create a word cloud from a list of texts with specified sentiment."""
        if not texts:
            logger.warning(f"No {sentiment} texts for word cloud")
            return None
        
        # Join all texts
        text = ' '.join(texts)
        
        if len(text.strip()) < 10:  # If text is too short
            logger.warning(f"Insufficient {sentiment} text for word cloud")
            return None
        
        # Set colors based on sentiment
        if sentiment == 'positive':
            colormap = 'Greens'
        elif sentiment == 'negative':
            colormap = 'Reds'
        else:
            colormap = 'Blues'
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                             colormap=colormap, max_words=100, contour_width=1,
                             contour_color='steelblue').generate(text)
        
        # Save the image
        clean_title = ''.join(c if c.isalnum() else '_' for c in video_title)
        clean_title = clean_title[:50]
        file_path = os.path.join(RESULTS_DIR, f"{clean_title}_{sentiment}_wordcloud.png")
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {sentiment} word cloud to {file_path}")
        return file_path
    
    # Separate comments by sentiment
    positive_texts = [r.get('comment_text', '') for r in results 
                     if isinstance(r.get('overall_sentiment'), str) and r.get('overall_sentiment') == 'positive']
    negative_texts = [r.get('comment_text', '') for r in results 
                     if isinstance(r.get('overall_sentiment'), str) and r.get('overall_sentiment') == 'negative']
    
    # Generate word clouds
    positive_cloud_path = create_word_cloud(positive_texts, 'positive')
    negative_cloud_path = create_word_cloud(negative_texts, 'negative')
    
    return positive_cloud_path, negative_cloud_path

def generate_charts(results, video_title, output_dir=None):
    """
    Generate all charts for the sentiment analysis results.
    
    Args:
        results (list): List of analysis results
        video_title (str): Title of the video
        output_dir (str): Optional output directory
        
    Returns:
        dict: Paths to the generated charts
    """
    # Override results directory if specified
    global RESULTS_DIR
    original_results_dir = RESULTS_DIR
    
    if output_dir:
        RESULTS_DIR = output_dir
        ensure_results_dir()
    
    try:
        # Generate all charts
        sentiment_pie_path = generate_sentiment_pie_chart(results, video_title)
        aspect_bar_path = generate_aspect_comparison_chart(results, video_title)
        positive_cloud_path, negative_cloud_path = generate_word_clouds(results, video_title)
        
        # Return paths - ensure we only return strings or None values
        chart_paths = {
            'sentiment_pie': sentiment_pie_path if sentiment_pie_path is None else os.path.basename(sentiment_pie_path),
            'aspect_bar': aspect_bar_path if aspect_bar_path is None else os.path.basename(aspect_bar_path),
            'positive_wordcloud': positive_cloud_path if positive_cloud_path is None else os.path.basename(positive_cloud_path),
            'negative_wordcloud': negative_cloud_path if negative_cloud_path is None else os.path.basename(negative_cloud_path)
        }
        
        logger.info(f"Generated all charts for '{video_title}'")
        return chart_paths
    
    except Exception as e:
        logger.error(f"Error generating charts: {str(e)}")
        # Return empty paths on error
        return {
            'sentiment_pie': None,
            'aspect_bar': None,
            'positive_wordcloud': None,
            'negative_wordcloud': None
        }
    
    finally:
        # Restore original results directory
        if output_dir:
            RESULTS_DIR = original_results_dir

def generate_full_report(results, video_title):
    """
    Generate a complete report with text summary and visualizations.
    
    Args:
        results (list): List of analysis results
        video_title (str): Title of the video
        
    Returns:
        dict: Paths to all generated report files
    """
    ensure_results_dir()
    
    # Generate text summary
    summary = generate_sentiment_summary(results, video_title)
    summary_path = save_sentiment_summary(summary, video_title)
    
    # Generate visualizations
    pie_chart_path = generate_sentiment_pie_chart(results, video_title)
    aspect_chart_path = generate_aspect_comparison_chart(results, video_title)
    wordcloud_paths = generate_word_clouds(results, video_title)
    
    # Return all paths
    report_paths = {
        'summary': summary_path,
        'pie_chart': pie_chart_path,
        'aspect_chart': aspect_chart_path,
        'wordclouds': wordcloud_paths
    }
    
    return report_paths

def print_report_summary(results, video_title):
    """
    Print a summary of the analysis results to the console.
    
    Args:
        results (list): List of analysis results
        video_title (str): Title of the video
    """
    summary = generate_sentiment_summary(results, video_title)
    print(summary)
    
    # Add information about generated files
    print("\nReport files have been saved to:")
    print(f"- {RESULTS_DIR}/")
    
    # Return the summary for possible further use
    return summary
