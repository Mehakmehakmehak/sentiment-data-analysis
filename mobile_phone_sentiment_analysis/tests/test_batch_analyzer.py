"""
Tests for batch analyzer functionality.
"""

import os
import sys
import unittest
from unittest.mock import patch, Mock, MagicMock
import tempfile

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mobile_phone_sentiment_analysis.batch_analyzer import (
    load_urls_from_file, 
    analyze_video, 
    batch_analyze_videos
)
from mobile_phone_sentiment_analysis.models.sentiment_analyzer import SentimentAnalyzer


class TestBatchAnalyzer(unittest.TestCase):
    """Tests for the batch analyzer module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file with sample URLs
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.temp_file.write("# Sample URLs\n")
        self.temp_file.write("https://www.youtube.com/watch?v=video1\n")
        self.temp_file.write("# Comment line\n")
        self.temp_file.write("https://www.youtube.com/watch?v=video2\n")
        self.temp_file.write("https://www.youtube.com/watch?v=video3\n")
        self.temp_file.close()
    
    def tearDown(self):
        """Tear down test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_load_urls_from_file(self):
        """Test loading URLs from a file."""
        urls = load_urls_from_file(self.temp_file.name)
        
        # Should find 3 URLs, skipping the comment lines
        self.assertEqual(len(urls), 3)
        self.assertEqual(urls[0], "https://www.youtube.com/watch?v=video1")
        self.assertEqual(urls[1], "https://www.youtube.com/watch?v=video2")
        self.assertEqual(urls[2], "https://www.youtube.com/watch?v=video3")
    
    @patch('mobile_phone_sentiment_analysis.batch_analyzer.collect_comments_from_video')
    @patch('mobile_phone_sentiment_analysis.batch_analyzer.generate_full_report')
    def test_analyze_video(self, mock_report, mock_collect):
        """Test analyzing a single video."""
        # Mock the comment collection
        mock_comments = [
            {
                'comment_id': 'id1',
                'author': 'Test User',
                'like_count': 5,
                'published_at': '2023-01-01T00:00:00Z',
                'comment_text': 'Great camera on this phone!'
            }
        ]
        mock_video_details = {
            'title': 'Test Video',
            'channel_title': 'Test Channel'
        }
        mock_csv_path = '/path/to/csv'
        
        mock_collect.return_value = (mock_comments, mock_video_details, mock_csv_path)
        
        # Mock the report generation
        mock_report.return_value = {
            'summary': '/path/to/summary',
            'charts': '/path/to/charts'
        }
        
        # Mock the sentiment analyzer
        mock_analyzer = MagicMock(spec=SentimentAnalyzer)
        mock_analyzer.analyze_sentiment.return_value = {
            'overall_sentiment': 'positive',
            'aspects': {'camera': 'positive'}
        }
        
        # Call the function
        results, details, reports = analyze_video(
            'https://www.youtube.com/watch?v=test',
            mock_analyzer,
            max_comments=10
        )
        
        # Check the results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['overall_sentiment'], 'positive')
        self.assertEqual(results[0]['aspects'], {'camera': 'positive'})
        self.assertEqual(results[0]['comment_id'], 'id1')
        
        # Verify mocks were called correctly
        mock_collect.assert_called_once_with(
            'https://www.youtube.com/watch?v=test',
            max_comments=10,
            output_dir='collected_comments'
        )
        mock_analyzer.analyze_sentiment.assert_called_once_with('Great camera on this phone!')
        mock_report.assert_called_once()
    
    @patch('mobile_phone_sentiment_analysis.batch_analyzer.analyze_video')
    @patch('mobile_phone_sentiment_analysis.batch_analyzer.load_urls_from_file')
    @patch('mobile_phone_sentiment_analysis.batch_analyzer.SentimentAnalyzer')
    def test_batch_analyze_videos(self, mock_analyzer_class, mock_load_urls, mock_analyze):
        """Test batch analyzing videos."""
        # Mock the URL loading
        mock_load_urls.return_value = [
            'https://www.youtube.com/watch?v=video1',
            'https://www.youtube.com/watch?v=video2'
        ]
        
        # Mock the analyzer
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock the video analysis
        mock_results = [{'overall_sentiment': 'positive'}]
        mock_details = {'title': 'Test Video'}
        mock_reports = {'summary': '/path/to/summary'}
        mock_analyze.return_value = (mock_results, mock_details, mock_reports)
        
        # Call the function
        results = batch_analyze_videos('urls.txt', max_comments=50)
        
        # Check the results
        self.assertEqual(len(results), 2)
        
        # Verify mocks were called correctly
        mock_load_urls.assert_called_once_with('urls.txt')
        self.assertEqual(mock_analyze.call_count, 2)
        
        # Check the arguments for the first call
        args, kwargs = mock_analyze.call_args_list[0]
        self.assertEqual(args[0], 'https://www.youtube.com/watch?v=video1')
        self.assertEqual(kwargs['max_comments'], 50)


if __name__ == '__main__':
    unittest.main() 