"""
Flask web application for mobile phone sentiment analysis.
"""

import os
import sys
import json
import logging
from flask import Flask, render_template, request, jsonify, url_for, redirect, session, send_from_directory
import argparse
import uuid

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from mobile_phone_sentiment_analysis.data.youtube_api import collect_comments_from_video
from mobile_phone_sentiment_analysis.models.sentiment_analyzer import SentimentAnalyzer
from mobile_phone_sentiment_analysis.utils.reporting import generate_charts

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Make sure to use absolute paths for static folder
app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.static_url_path = '/static'

# Set up folder configuration with absolute paths
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(app.static_folder, 'results')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('webapp.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize sentiment analyzer
analyzer = None

# In-memory storage for analysis results
analysis_store = {}

def get_analyzer():
    """Get or initialize the sentiment analyzer."""
    global analyzer
    if analyzer is None:
        logger.info("Initializing sentiment analyzer...")
        analyzer = SentimentAnalyzer()
        try:
            analyzer.load_models()
            logger.info("Loaded existing sentiment models")
        except Exception as e:
            logger.warning(f"Could not load models, using rule-based analysis only: {str(e)}")
    return analyzer

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze comments from a YouTube video URL.
    
    Returns:
        Rendered results page or error page.
    """
    video_url = request.form.get('video_url')
    max_comments = int(request.form.get('max_comments', 100))
    
    if not video_url:
        return render_template('index.html', error="Please enter a YouTube video URL")
    
    try:
        # Get sentiment analyzer
        sentiment_analyzer = get_analyzer()
        
        # Collect comments
        logger.info(f"Collecting comments from {video_url}")
        comments, video_details, csv_path = collect_comments_from_video(
            video_url, max_comments=max_comments, output_dir='collected_comments'
        )
        
        if not comments:
            return render_template('index.html', 
                                  error="No comments found for this video. Please try another video.")
        
        # Analyze comments
        logger.info(f"Analyzing {len(comments)} comments")
        analysis_results = []
        
        for comment in comments:
            try:
                result = sentiment_analyzer.analyze_sentiment(comment['comment_text'])
                
                # Ensure overall sentiment is a string
                if not isinstance(result['overall_sentiment'], str):
                    result['overall_sentiment'] = str(result['overall_sentiment'])
                
                # Ensure all aspect sentiments are strings
                for aspect, sentiment in result.get('aspects', {}).items():
                    if not isinstance(sentiment, str):
                        result['aspects'][aspect] = str(sentiment)
                
                # Add comment metadata
                result['comment_id'] = comment['comment_id']
                result['author'] = comment['author']
                result['like_count'] = comment['like_count']
                result['published_at'] = comment['published_at']
                result['comment_text'] = comment['comment_text']
                
                analysis_results.append(result)
            except Exception as comment_error:
                logger.error(f"Error analyzing comment {comment.get('comment_id', 'unknown')}: {str(comment_error)}")
                # Skip failed comments but continue with the rest
                continue
        
        if not analysis_results:
            return render_template('index.html', 
                                  error="Failed to analyze comments. Please try another video.")
        
        # Generate visualizations
        logger.info("Generating visualizations")
        
        # Make sure to use the absolute path to the results folder
        results_folder = app.config['RESULTS_FOLDER']
        
        # Generate charts with the absolute path
        try:
            chart_paths = generate_charts(analysis_results, video_details['title'], 
                                        output_dir=results_folder)
        except Exception as chart_error:
            logger.error(f"Error generating charts: {str(chart_error)}")
            # Provide empty chart paths if generation fails
            chart_paths = {
                'sentiment_pie': None,
                'aspect_bar': None,
                'positive_wordcloud': None,
                'negative_wordcloud': None
            }
        
        # Generate a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Store results in server memory instead of session
        analysis_store[analysis_id] = {
            'analysis_results': analysis_results,
            'video_details': video_details,
            'chart_paths': chart_paths
        }
        
        # Store only the ID in the session
        session['analysis_id'] = analysis_id
        
        # Redirect to results page
        return redirect(url_for('results'))
    
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return render_template('index.html', 
                              error=f"An error occurred: {str(e)}. Please try again.")

@app.route('/results')
def results():
    """Render the results page."""
    # Get analysis ID from session
    analysis_id = session.get('analysis_id')
    
    if not analysis_id or analysis_id not in analysis_store:
        return redirect(url_for('index'))
    
    # Get results from server memory
    stored_data = analysis_store[analysis_id]
    analysis_results = stored_data['analysis_results']
    video_details = stored_data['video_details']
    chart_paths = stored_data['chart_paths']
    
    # Count sentiments (ensuring all values are strings)
    sentiments = []
    for r in analysis_results:
        sentiment = r.get('overall_sentiment', 'neutral')
        # Convert non-string sentiments to strings
        if not isinstance(sentiment, str):
            sentiment = str(sentiment)
        sentiments.append(sentiment)
    
    sentiment_counts = {
        'positive': sentiments.count('positive'),
        'negative': sentiments.count('negative'),
        'neutral': sentiments.count('neutral')
    }
    
    # Count aspects
    aspects = {}
    for result in analysis_results:
        for aspect, sentiment in result.get('aspects', {}).items():
            if aspect not in aspects:
                aspects[aspect] = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            # Handle case where sentiment might be a dictionary or other non-string type
            if isinstance(sentiment, dict):
                # Skip it or handle dictionary sentiment (depends on structure)
                continue
            elif not isinstance(sentiment, str):
                # Convert to string if it's not already
                sentiment = str(sentiment)
            
            # Only increment if sentiment is one of the expected values
            if sentiment in aspects[aspect]:
                aspects[aspect][sentiment] += 1
    
    # Calculate percentages
    total_comments = len(analysis_results)
    sentiment_percentages = {
        'positive': round((sentiment_counts['positive'] / total_comments) * 100, 1) if total_comments > 0 else 0,
        'negative': round((sentiment_counts['negative'] / total_comments) * 100, 1) if total_comments > 0 else 0,
        'neutral': round((sentiment_counts['neutral'] / total_comments) * 100, 1) if total_comments > 0 else 0
    }
    
    # Adjust chart paths to be relative to static
    for key, path in chart_paths.items():
        if path:
            # Get just the filename without the full path
            filename = os.path.basename(path)
            # Store just the filename, not the full path
            chart_paths[key] = filename
    
    return render_template(
        'results.html',
        video_details=video_details,
        sentiment_counts=sentiment_counts,
        sentiment_percentages=sentiment_percentages,
        aspects=aspects,
        chart_paths=chart_paths,
        total_comments=total_comments
    )

@app.route('/comments')
def comments():
    """
    Get filtered comments based on query parameters.
    
    Query parameters:
        sentiment: Filter by overall sentiment (positive, negative, neutral)
        aspect: Filter by aspect (camera, battery, etc.)
        aspect_sentiment: Filter by aspect sentiment (positive, negative, neutral)
        
    Returns:
        Rendered comments page.
    """
    # Get filters from query parameters
    sentiment_filter = request.args.get('sentiment', 'all')
    aspect_filter = request.args.get('aspect', 'all')
    aspect_sentiment_filter = request.args.get('aspect_sentiment', 'all')
    
    # Get analysis ID from session
    analysis_id = session.get('analysis_id')
    
    if not analysis_id or analysis_id not in analysis_store:
        return redirect(url_for('index'))
    
    # Get results from server memory
    stored_data = analysis_store[analysis_id]
    analysis_results = stored_data['analysis_results']
    video_details = stored_data['video_details']
    
    # Filter comments
    filtered_comments = []
    
    for result in analysis_results:
        try:
            # Ensure overall_sentiment is a string
            overall_sentiment = result.get('overall_sentiment', 'neutral')
            if not isinstance(overall_sentiment, str):
                overall_sentiment = str(overall_sentiment)
                result['overall_sentiment'] = overall_sentiment
            
            # First filter by overall sentiment if specified
            if sentiment_filter != 'all' and overall_sentiment != sentiment_filter:
                continue
            
            # Then filter by aspect if specified
            if aspect_filter != 'all':
                # Skip comments that don't mention the aspect
                if aspect_filter not in result.get('aspects', {}):
                    continue
                
                # Get the sentiment for this aspect
                aspect_sentiment = result['aspects'][aspect_filter]
                
                # Handle case where sentiment might be a dictionary or other non-string type
                if isinstance(aspect_sentiment, dict):
                    # Skip it or handle dictionary sentiment
                    continue
                elif not isinstance(aspect_sentiment, str):
                    # Convert to string if it's not already
                    aspect_sentiment = str(aspect_sentiment)
                    result['aspects'][aspect_filter] = aspect_sentiment
                
                # Filter by aspect sentiment if specified
                if aspect_sentiment_filter != 'all':
                    if aspect_sentiment != aspect_sentiment_filter:
                        continue
            
            # Add comment to filtered list
            filtered_comments.append(result)
        except Exception as e:
            logger.error(f"Error filtering comment: {str(e)}")
            continue
    
    # List of available aspects for the dropdown
    all_aspects = set()
    for result in analysis_results:
        for aspect in result.get('aspects', {}):
            all_aspects.add(aspect)
    
    return render_template(
        'comments.html',
        video_details=video_details,
        comments=filtered_comments,
        total_filtered=len(filtered_comments),
        total_comments=len(analysis_results),
        selected_sentiment=sentiment_filter,
        selected_aspect=aspect_filter,
        selected_aspect_sentiment=aspect_sentiment_filter,
        all_aspects=sorted(all_aspects)
    )

@app.route('/api/comments')
def api_comments():
    """API endpoint to get filtered comments in JSON format."""
    # Get filters from query parameters
    sentiment_filter = request.args.get('sentiment', 'all')
    aspect_filter = request.args.get('aspect', 'all')
    aspect_sentiment_filter = request.args.get('aspect_sentiment', 'all')
    
    # Get analysis ID from session
    analysis_id = session.get('analysis_id')
    
    if not analysis_id or analysis_id not in analysis_store:
        return jsonify({'error': 'No analysis results found'})
    
    # Get results from server memory
    stored_data = analysis_store[analysis_id]
    analysis_results = stored_data['analysis_results']
    
    # Filter comments
    filtered_comments = []
    
    for result in analysis_results:
        try:
            # Ensure overall_sentiment is a string
            overall_sentiment = result.get('overall_sentiment', 'neutral')
            if not isinstance(overall_sentiment, str):
                overall_sentiment = str(overall_sentiment)
                result['overall_sentiment'] = overall_sentiment
                
            # First filter by overall sentiment if specified
            if sentiment_filter != 'all' and overall_sentiment != sentiment_filter:
                continue
            
            # Then filter by aspect if specified
            if aspect_filter != 'all':
                # Skip comments that don't mention the aspect
                if aspect_filter not in result.get('aspects', {}):
                    continue
                
                # Get the sentiment for this aspect
                aspect_sentiment = result['aspects'][aspect_filter]
                
                # Handle case where sentiment might be a dictionary or other non-string type
                if isinstance(aspect_sentiment, dict):
                    # Skip it or handle dictionary sentiment
                    continue
                elif not isinstance(aspect_sentiment, str):
                    # Convert to string if it's not already
                    aspect_sentiment = str(aspect_sentiment)
                    result['aspects'][aspect_filter] = aspect_sentiment
                
                # Filter by aspect sentiment if specified
                if aspect_sentiment_filter != 'all':
                    if aspect_sentiment != aspect_sentiment_filter:
                        continue
            
            # Add comment to filtered list
            filtered_comments.append(result)
        except Exception as e:
            logger.error(f"Error filtering comment for API: {str(e)}")
            continue
    
    return jsonify({'comments': filtered_comments})

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    API endpoint to analyze a single comment.
    
    Expects JSON with a 'comment' field.
    
    Returns:
        JSON response with sentiment analysis results.
    """
    try:
        data = request.get_json()
        
        if not data or 'comment' not in data:
            return jsonify({
                'error': 'Invalid request. Please provide a comment field.'
            }), 400
            
        comment_text = data['comment']
        
        # Get sentiment analyzer
        sentiment_analyzer = get_analyzer()
        
        # Analyze comment
        result = sentiment_analyzer.analyze_sentiment(comment_text)
        
        # Ensure all values are JSON serializable
        for aspect, data in result.get('aspects', {}).items():
            if isinstance(data, dict):
                # Keep the dict structure but ensure values are serializable
                for k, v in data.items():
                    if not isinstance(v, (str, int, float, bool, type(None))):
                        result['aspects'][aspect][k] = str(v)
            else:
                # Convert the value to a dictionary with sentiment
                result['aspects'][aspect] = {'sentiment': str(data)}
        
        # Add the original comment
        result['comment'] = comment_text
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'error': f"An error occurred: {str(e)}"
        }), 500

@app.template_filter('format_date')
def format_date(date_string):
    """Format ISO date strings for display."""
    if not date_string:
        return ""
    try:
        from datetime import datetime
        dt = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return date_string

@app.route('/static/results/<path:filename>')
def custom_static(filename):
    """
    Directly serve files from the results directory.
    This is a fallback in case the standard static serving doesn't work.
    """
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the mobile phone sentiment analysis web app')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    
    port = int(os.environ.get("PORT", args.port))
    app.run(host="0.0.0.0", port=port, debug=True) 