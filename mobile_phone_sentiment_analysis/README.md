# Mobile Phone Sentiment Analysis System

A specialized sentiment analysis system focused on extracting sentiment from YouTube comments about mobile phone reviews. The system analyzes both overall sentiment and specific aspects of phones (camera, battery, performance, display, design, and price).

## Features

- **YouTube Comment Collection**: Collect comments from mobile phone review videos using the YouTube Data API
- **Hybrid Sentiment Analysis**: Combines machine learning and rule-based approaches for more accurate results
- **Aspect-Based Sentiment**: Identifies how people feel about specific phone features
- **Multi-Level Analysis**: Analyzes sentiment at both comment and feature levels
- **Comprehensive Reports**: Generates text summaries and visualizations of analysis results
- **Batch Processing**: Analyze multiple videos in one run with aggregated reporting

## System Architecture

The system consists of several components:

1. **Data Collection**: Uses the YouTube Data API to collect comments from phone review videos
2. **Data Preprocessing**: Cleans and processes text data for analysis
3. **Feature Extraction**: Identifies aspects (camera, battery, etc.) mentioned in comments
4. **Sentiment Analysis**: Uses both ML models and rules to determine sentiment
5. **Reporting**: Generates summaries and visualizations of analysis results

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mobile_phone_sentiment_analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data and spaCy model:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); import spacy; spacy.cli.download('en_core_web_sm')"
```

4. Set up YouTube API credentials:
   - Create a project in the [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the YouTube Data API v3
   - Create API key credentials
   - Set the API key in your environment variables:
   ```bash
   export YOUTUBE_API_KEY=your_api_key_here
   ```

## Usage

### Training Models

To train sentiment analysis models using the annotated dataset:

```bash
python main.py train
```

To force retraining of models (even if models already exist):

```bash
python main.py train --force
```

You can also use the dedicated training script:
```bash
python train_models.py --force
```

### Web Application

The system includes a Flask web application for easy interaction:

```bash
# Start the web application
python app.py
```

Then open your browser and navigate to http://localhost:5000

The web application allows you to:
- Paste YouTube video URLs and analyze comments
- View visualization charts for sentiment analysis
- Filter comments by sentiment and aspects
- Explore detailed aspect-based sentiment breakdown

### Analyzing a Single Video

To analyze comments from a specific YouTube video:

```bash
python main.py analyze --url <youtube-video-url> --max <max-comments>
```

Example:
```bash
python main.py analyze --url https://www.youtube.com/watch?v=dQw4w9WgXcQ --max 200
```

### Batch Analysis

The system provides two ways to perform batch analysis:

#### 1. Using main.py

To analyze multiple videos from a file (one URL per line):

```bash
python main.py batch --file <url-file> --max <max-comments-per-video>
```

Example:
```bash
python main.py batch --file sample_urls.txt --max 100
```

#### 2. Using the dedicated batch analyzer

For more advanced batch processing with detailed summaries:

```bash
python batch_analyzer.py --file <url-file> --max <max-comments-per-video>
```

Example:
```bash
python batch_analyzer.py --file sample_urls.txt --max 150
```

### Running Examples

The repository includes example scripts to help you get started:

```bash
# Run analysis on sample comments
python example.py
```

## URL File Format

For batch processing, create a text file with YouTube URLs, one per line:

```
# iPhone reviews
https://www.youtube.com/watch?v=FT3ODSg1GFE

# Samsung Galaxy reviews  
https://www.youtube.com/watch?v=yCL5hHLNtBQ
```

Lines starting with `#` are treated as comments and ignored. A sample file is provided at `sample_urls.txt`.

## Testing

The project includes unit tests to verify functionality. To run the tests:

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py -v

# Run a specific test file
python run_tests.py tests/test_batch_analyzer.py

# Run a specific test module
python run_tests.py tests.test_batch_analyzer
```

## Dataset

The system includes an annotated dataset of mobile phone review comments, with labels for:

- Overall sentiment (positive, negative, neutral)
- Whether specific aspects (camera, battery, etc.) are mentioned
- Sentiment for each mentioned aspect

## Output

Analysis results are saved to the following directories:

- `mobile_phone_sentiment_analysis/results/`: Individual analysis reports
- `collected_comments/`: CSV files with raw comments
- `batch_results/`: Summary reports for batch analysis

Each analysis generates:

- Text summary of sentiment analysis
- Pie chart of overall sentiment distribution
- Bar chart comparing sentiment across different aspects
- Word clouds for positive and negative comments

Batch analysis additionally produces:
- A CSV file with aggregated statistics for all videos
- A text summary report with key metrics for each video

## Configuration

The system's behavior can be configured by modifying settings in `config/config.py`, including:

- YouTube API key and data collection parameters
- NLP settings (stopwords, aspect keywords, sentiment lexicons)
- Model parameters (features, estimators, etc.)

## License

[MIT License](LICENSE) 