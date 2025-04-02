"""
YouTube Data API interaction module for collecting comments.
"""

import sys
import os
import time
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
import re

# Import configuration
sys.path.append(".")
from mobile_phone_sentiment_analysis.config.config import YOUTUBE_API_KEY, MAX_COMMENTS_PER_VIDEO

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_youtube_api():
    """
    Build and return a YouTube API client.
    
    Returns:
        object: YouTube API client
    """
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        logger.info("Successfully built YouTube API client")
        return youtube
    except Exception as e:
        logger.error(f"Failed to build YouTube API client: {str(e)}")
        return None

def extract_video_id(url):
    """
    Extract the video ID from a YouTube URL.
    
    Args:
        url (str): YouTube URL
    
    Returns:
        str: YouTube video ID or None if not found
    """
    if not url:
        return None
    
    # Patterns for YouTube URLs
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&\s]+)',  # Standard watch URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^\?\s]+)',    # Embed URL
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^\?\s]+)'              # Short URL
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def get_video_details(youtube, video_id):
    """
    Get details for a YouTube video.
    
    Args:
        youtube: YouTube API client
        video_id (str): YouTube video ID
    
    Returns:
        dict: Video details including title, channel, etc.
    """
    if not youtube or not video_id:
        return None
    
    try:
        response = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()
        
        if 'items' not in response or not response['items']:
            logger.warning(f"No video found with ID: {video_id}")
            return None
        
        video_info = response['items'][0]
        
        details = {
            'video_id': video_id,
            'title': video_info['snippet']['title'],
            'channel_title': video_info['snippet']['channelTitle'],
            'published_at': video_info['snippet']['publishedAt'],
            'view_count': video_info['statistics'].get('viewCount', 0),
            'like_count': video_info['statistics'].get('likeCount', 0),
            'comment_count': video_info['statistics'].get('commentCount', 0)
        }
        
        logger.info(f"Retrieved details for video: {details['title']}")
        return details
    
    except HttpError as e:
        logger.error(f"HTTP error when getting video details: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error when getting video details: {str(e)}")
        return None

def get_video_comments(youtube, video_id, max_comments=None):
    """
    Get comments for a YouTube video.
    
    Args:
        youtube: YouTube API client
        video_id (str): YouTube video ID
        max_comments (int): Maximum number of comments to retrieve
        
    Returns:
        list: List of comment dictionaries
    """
    if not youtube or not video_id:
        return []
    
    if max_comments is None:
        max_comments = MAX_COMMENTS_PER_VIDEO
    
    try:
        comments = []
        next_page_token = None
        
        # Loop until we have enough comments or there are no more pages
        while len(comments) < max_comments:
            # Prepare the request
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),  # API limit is 100 per request
                pageToken=next_page_token,
                textFormat='plainText'
            )
            
            # Execute the request
            response = request.execute()
            
            # Process the comments
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                
                comment_info = {
                    'comment_id': item['id'],
                    'comment_text': comment['textDisplay'],
                    'author': comment['authorDisplayName'],
                    'like_count': comment['likeCount'],
                    'published_at': comment['publishedAt']
                }
                
                comments.append(comment_info)
                
                if len(comments) >= max_comments:
                    break
            
            # Check if there are more pages
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
            
            # Be nice to the API - add a small delay between requests
            time.sleep(0.5)
        
        logger.info(f"Retrieved {len(comments)} comments for video ID: {video_id}")
        return comments
    
    except HttpError as e:
        logger.error(f"HTTP error when getting comments: {str(e)}")
        if "commentsDisabled" in str(e):
            logger.warning("Comments are disabled for this video")
        return []
    except Exception as e:
        logger.error(f"Error when getting comments: {str(e)}")
        return []

def save_comments_to_csv(comments, video_details, output_dir='.'):
    """
    Save comments to a CSV file.
    
    Args:
        comments (list): List of comment dictionaries
        video_details (dict): Video details dictionary
        output_dir (str): Directory to save the CSV file
        
    Returns:
        str: Path to the saved CSV file
    """
    if not comments or not video_details:
        logger.warning("No comments or video details to save")
        return None
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame from comments
        df = pd.DataFrame(comments)
        
        # Clean up video title for use as a filename
        video_title = video_details['title']
        clean_title = ''.join(c if c.isalnum() else '_' for c in video_title)
        clean_title = clean_title[:50]  # Truncate if too long
        
        # Define file path
        file_path = os.path.join(output_dir, f"{clean_title}_Comments.csv")
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        
        logger.info(f"Saved {len(comments)} comments to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving comments to CSV: {str(e)}")
        return None

def collect_comments_from_video(video_url, max_comments=None, output_dir='.'):
    """
    Collect comments from a YouTube video and save to CSV.
    
    Args:
        video_url (str): YouTube video URL
        max_comments (int): Maximum number of comments to collect
        output_dir (str): Directory to save the CSV file
        
    Returns:
        tuple: (comments list, video details dict, CSV file path)
    """
    # Extract video ID from URL
    video_id = extract_video_id(video_url)
    if not video_id:
        logger.error(f"Could not extract video ID from URL: {video_url}")
        return None, None, None
    
    # Build YouTube API client
    youtube = build_youtube_api()
    if not youtube:
        return None, None, None
    
    # Get video details
    video_details = get_video_details(youtube, video_id)
    if not video_details:
        return None, None, None
    
    # Get comments
    comments = get_video_comments(youtube, video_id, max_comments)
    
    # Save comments to CSV
    csv_path = save_comments_to_csv(comments, video_details, output_dir)
    
    return comments, video_details, csv_path
