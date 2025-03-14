import os
import pandas as pd
import googleapiclient.discovery
from googleapiclient.errors import HttpError
from tqdm import tqdm
import time
import sys
import json

# Add project root to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config.config import (
    YOUTUBE_API_KEY,
    YOUTUBE_API_SERVICE_NAME,
    YOUTUBE_API_VERSION,
    MAX_COMMENTS_PER_VIDEO,
    RAW_DATA_DIR
)


def create_youtube_client():
    """Creates and returns a YouTube API client."""
    return googleapiclient.discovery.build(
        YOUTUBE_API_SERVICE_NAME,
        YOUTUBE_API_VERSION,
        developerKey=YOUTUBE_API_KEY
    )


def get_video_comments(youtube_client, video_id, max_comments=MAX_COMMENTS_PER_VIDEO):
    """
    Fetches comments for a specific YouTube video.

    Args:
        youtube_client: YouTube API client
        video_id: YouTube video ID
        max_comments: Maximum number of comments to fetch

    Returns:
        List of comment dictionaries
    """
    comments = []
    next_page_token = None

    try:
        while len(comments) < max_comments:
            # Make API request for comments
            response = youtube_client.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),  # YouTube API limit is 100 per request
                pageToken=next_page_token if next_page_token else "",
                textFormat="plainText"
            ).execute()

            # Process comments
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']

                # Extract relevant fields
                comment_info = {
                    'comment_id': item['id'],
                    'video_id': video_id,
                    'author': comment['authorDisplayName'],
                    'text': comment['textDisplay'],
                    'published_at': comment['publishedAt'],
                    'like_count': comment['likeCount'],
                    'reply_count': item['snippet']['totalReplyCount']
                }

                comments.append(comment_info)

                # Get replies if they exist (limited to first page of replies)
                if 'replies' in item and item['snippet']['totalReplyCount'] > 0:
                    for reply in item['replies']['comments']:
                        reply_snippet = reply['snippet']
                        reply_info = {
                            'comment_id': reply['id'],
                            'video_id': video_id,
                            'author': reply_snippet['authorDisplayName'],
                            'text': reply_snippet['textDisplay'],
                            'published_at': reply_snippet['publishedAt'],
                            'like_count': reply_snippet['likeCount'],
                            'reply_count': 0,  # Replies don't have replies
                            'parent_id': item['id']  # Reference to parent comment
                        }
                        comments.append(reply_info)

            # Check if there are more comments
            next_page_token = response.get('nextPageToken')
            if not next_page_token or len(comments) >= max_comments:
                break

            # Respect YouTube API quotas with a short delay
            time.sleep(0.5)

    except HttpError as e:
        print(f"An HTTP error occurred: {e}")

    return comments[:max_comments]


def get_phone_review_videos(youtube_client, search_query, max_results=50):
    """
    Searches for mobile phone review videos.

    Args:
        youtube_client: YouTube API client
        search_query: Search query string (e.g., "iPhone 15 Pro review")
        max_results: Maximum number of results to return

    Returns:
        List of video dictionaries
    """
    try:
        search_response = youtube_client.search().list(
            q=search_query,
            part="id,snippet",
            maxResults=max_results,
            type="video",
            videoDefinition="high",
            relevanceLanguage="en"
        ).execute()

        videos = []
        for item in search_response['items']:
            if item['id']['kind'] == 'youtube#video':
                video_info = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'channel_id': item['snippet']['channelId'],
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'description': item['snippet']['description']
                }
                videos.append(video_info)

        return videos

    except HttpError as e:
        print(f"An HTTP error occurred: {e}")
        return []


def collect_comments_for_phone_model(phone_model, max_videos=5, max_comments_per_video=200):
    """
    Collects comments for a specific phone model.

    Args:
        phone_model: Phone model name (e.g., "iPhone 15 Pro")
        max_videos: Maximum number of videos to process
        max_comments_per_video: Maximum comments to collect per video

    Returns:
        DataFrame with all collected comments
    """
    youtube_client = create_youtube_client()
    search_query = f"{phone_model} review"

    print(f"Searching for videos: {search_query}")
    videos = get_phone_review_videos(youtube_client, search_query, max_videos)

    all_comments = []
    for video in tqdm(videos, desc="Processing videos"):
        print(f"Collecting comments for video: {video['title']} (ID: {video['video_id']})")
        video_comments = get_video_comments(
            youtube_client,
            video['video_id'],
            max_comments_per_video
        )

        # Add video metadata to each comment
        for comment in video_comments:
            comment['video_title'] = video['title']
            comment['channel_title'] = video['channel_title']
            comment['phone_model'] = phone_model

        all_comments.extend(video_comments)
        print(f"Collected {len(video_comments)} comments")

    # Convert to DataFrame
    comments_df = pd.DataFrame(all_comments)

    # Save to file
    output_file = os.path.join(RAW_DATA_DIR, f"{phone_model.replace(' ', '_')}_comments.csv")
    comments_df.to_csv(output_file, index=False)
    print(f"Saved {len(comments_df)} comments to {output_file}")

    # Also save as JSON for backup
    json_file = os.path.join(RAW_DATA_DIR, f"{phone_model.replace(' ', '_')}_comments.json")
    comments_df.to_json(json_file, orient='records')

    return comments_df


def main():
    """
    Main function to collect comments for multiple phone models.
    """
    phone_models = [
        "iPhone 15 Pro",
        "Samsung Galaxy S23 Ultra",
        "Google Pixel 8 Pro",
        "OnePlus 12"
    ]

    for phone_model in phone_models:
        print(f"\n{'=' * 50}\nCollecting data for {phone_model}\n{'=' * 50}")
        collect_comments_for_phone_model(
            phone_model,
            max_videos=3,  # Limiting for initial testing
            max_comments_per_video=100  # Limiting for initial testing
        )


if __name__ == "__main__":
    main()