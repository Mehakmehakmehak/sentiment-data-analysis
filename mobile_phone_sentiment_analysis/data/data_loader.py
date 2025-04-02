"""
Data loading and processing utilities for the annotated dataset.
"""

import sys
import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import logging

# Import configuration
sys.path.append(".")
from mobile_phone_sentiment_analysis.config.config import ANNOTATED_DATA_DIR, TEST_SIZE, RANDOM_STATE

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_annotated_file(file_path):
    """
    Load a single annotated CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the annotated data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded annotated file: {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        return None

def load_all_annotated_data(annotated_dir=None):
    """
    Load and combine all annotated CSV files in the specified directory.
    
    Args:
        annotated_dir (str): Directory containing annotated CSV files
        
    Returns:
        pd.DataFrame: Combined DataFrame with all annotated data
    """
    if annotated_dir is None:
        annotated_dir = ANNOTATED_DATA_DIR
    
    try:
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(annotated_dir, '*.csv'))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {annotated_dir}")
            return None
        
        # Load each file
        dfs = []
        for file_path in csv_files:
            df = load_annotated_file(file_path)
            if df is not None:
                # Add source file information
                df['source_file'] = os.path.basename(file_path)
                dfs.append(df)
        
        if not dfs:
            logger.warning("No valid data frames loaded")
            return None
        
        # Combine all data frames
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(dfs)} files into DataFrame with shape {combined_df.shape}")
        
        return combined_df
    
    except Exception as e:
        logger.error(f"Error loading annotated data: {str(e)}")
        return None

def clean_annotated_data(df):
    """
    Clean and preprocess the annotated data.
    
    Args:
        df (pd.DataFrame): Annotated data DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if df is None or df.empty:
        return None
    
    try:
        # Make a copy to avoid modifying the original
        clean_df = df.copy()
        
        # Drop rows with missing comment text
        clean_df = clean_df.dropna(subset=['comment_text'])
        
        # Remove duplicates
        clean_df = clean_df.drop_duplicates(subset=['comment_text'])
        
        # Clean up the comment text (remove quotation marks used in the CSV)
        clean_df['comment_text'] = clean_df['comment_text'].str.replace('^"|"$', '', regex=True)
        
        # Clean overall sentiment - keep only valid sentiment labels
        valid_sentiments = ['positive', 'negative', 'neutral']
        
        # Replace non-standard sentiments with 'neutral'
        if 'overall_sentiment' in clean_df.columns:
            # First convert to string
            clean_df['overall_sentiment'] = clean_df['overall_sentiment'].astype(str)
            
            # Replace non-standard values
            clean_df.loc[~clean_df['overall_sentiment'].isin(valid_sentiments), 'overall_sentiment'] = 'neutral'
        
        # Clean aspect sentiments
        sentiment_cols = [col for col in clean_df.columns if col.endswith('_sentiment')]
        for col in sentiment_cols:
            if col in clean_df.columns:
                # First convert to string
                clean_df[col] = clean_df[col].astype(str)
                
                # Replace any non-standard values with 'neutral'
                clean_df.loc[~clean_df[col].isin(valid_sentiments), col] = 'neutral'
        
        # Make sure the feature_mentioned columns are boolean
        mention_cols = [col for col in clean_df.columns if col.endswith('_mentioned')]
        for col in mention_cols:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].fillna(False).astype(bool)
        
        logger.info(f"Cleaned annotated data. Original shape: {df.shape}, New shape: {clean_df.shape}")
        return clean_df
    
    except Exception as e:
        logger.error(f"Error cleaning annotated data: {str(e)}")
        return df

def split_data_by_sentiment(df, text_col='comment_text', sentiment_col='overall_sentiment', test_size=None):
    """
    Split data into train and test sets, maintaining sentiment distribution.
    
    Args:
        df (pd.DataFrame): DataFrame to split
        text_col (str): Column containing the text data
        sentiment_col (str): Column containing the sentiment labels
        test_size (float): Proportion of data to use for testing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if df is None or df.empty:
        return None, None, None, None
    
    if test_size is None:
        test_size = TEST_SIZE
    
    try:
        # Ensure the required columns exist
        if text_col not in df.columns or sentiment_col not in df.columns:
            logger.error(f"Required columns ({text_col}, {sentiment_col}) not found in DataFrame")
            return None, None, None, None
        
        # Split the data
        X = df[text_col]
        y = df[sentiment_col]
        
        # Check sentiment distribution
        sentiment_counts = y.value_counts()
        logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
        
        # If any sentiment class has < 2 examples, we can't use stratify
        if sentiment_counts.min() < 2:
            logger.warning("Some sentiment classes have too few examples for stratified sampling. Using random split instead.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE
            )
        else:
            # Use stratified sampling
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
            )
        
        logger.info(f"Split data: {len(X_train)} training samples, {len(X_test)} testing samples")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        # Fallback to simple random split
        logger.warning("Falling back to simple random split without stratification")
        try:
            X = df[text_col]
            y = df[sentiment_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE
            )
            return X_train, X_test, y_train, y_test
        except Exception as e2:
            logger.error(f"Error in fallback split: {str(e2)}")
            return None, None, None, None

def create_aspect_datasets(df):
    """
    Create separate datasets for each aspect mentioned in the annotated data.
    
    Args:
        df (pd.DataFrame): Cleaned annotated DataFrame
        
    Returns:
        dict: Dictionary mapping aspects to DataFrames
    """
    if df is None or df.empty:
        return {}
    
    aspect_datasets = {}
    aspects = ['camera', 'battery', 'performance', 'display', 'design', 'price']
    
    try:
        for aspect in aspects:
            # Check for aspect mention column
            mention_col = f"{aspect}_mentioned"
            sentiment_col = f"{aspect}_sentiment"
            
            # Check if required columns exist
            if mention_col in df.columns and sentiment_col in df.columns:
                # Get rows where the aspect is mentioned
                aspect_df = df[df[mention_col] == True].copy()
                
                if len(aspect_df) > 0:
                    # Keep the original comment_text and aspect_sentiment column names
                    aspect_df = aspect_df[['comment_text', sentiment_col]].copy()
                    
                    # Ensure sentiment values are strings
                    aspect_df[sentiment_col] = aspect_df[sentiment_col].astype(str)
                    
                    # Store in dictionary
                    aspect_datasets[aspect] = aspect_df
                    logger.info(f"Created dataset for {aspect} with {len(aspect_df)} samples")
            else:
                # Use rule-based detection as fallback
                # Extract comments that mention the aspect based on keywords
                aspect_keywords = {
                    'camera': ['camera', 'photo', 'picture', 'image', 'photography', 'video', 'selfie', 'lens'],
                    'battery': ['battery', 'charging', 'power', 'drain', 'life', 'backup'],
                    'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'processing', 'snappy', 'processor'],
                    'display': ['display', 'screen', 'resolution', 'bright', 'color', 'refresh'],
                    'design': ['design', 'build', 'feel', 'body', 'glass', 'metal', 'plastic', 'premium', 'look'],
                    'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'affordable', 'overpriced']
                }
                
                # Get keywords for this aspect
                keywords = aspect_keywords.get(aspect, [aspect])
                
                # Find comments that mention any of the keywords
                aspect_df = df[df['comment_text'].str.lower().str.contains('|'.join(keywords), na=False)].copy()
                
                if len(aspect_df) > 0:
                    # Use overall sentiment as a proxy when aspect-specific sentiment is not available
                    aspect_df = aspect_df[['comment_text', 'overall_sentiment']].copy()
                    
                    # Ensure sentiment values are strings
                    aspect_df['overall_sentiment'] = aspect_df['overall_sentiment'].astype(str)
                    
                    # Rename only the sentiment column to match the expected format
                    aspect_df = aspect_df.rename(columns={'overall_sentiment': f'{aspect}_sentiment'})
                    
                    # Store in dictionary
                    aspect_datasets[aspect] = aspect_df
                    logger.info(f"Created fallback dataset for {aspect} with {len(aspect_df)} samples")
        
        return aspect_datasets
    
    except Exception as e:
        logger.error(f"Error creating aspect datasets: {str(e)}")
        return {}

def get_processed_data():
    """
    Load, clean, and process all annotated data.
    
    Returns:
        tuple: (cleaned_df, train_test_splits, aspect_datasets)
    """
    # Load all annotated data
    df = load_all_annotated_data()
    
    if df is None or df.empty:
        return None, None, None
    
    # Clean the data
    cleaned_df = clean_annotated_data(df)
    
    if cleaned_df is None or cleaned_df.empty:
        return None, None, None
    
    # Split for overall sentiment analysis
    X_train, X_test, y_train, y_test = split_data_by_sentiment(cleaned_df)
    
    train_test_splits = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Create aspect-specific datasets
    aspect_datasets = create_aspect_datasets(cleaned_df)
    
    return cleaned_df, train_test_splits, aspect_datasets
