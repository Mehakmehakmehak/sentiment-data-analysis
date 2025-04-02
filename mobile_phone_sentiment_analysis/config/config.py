"""
Configuration settings for the mobile phone sentiment analysis system.
"""

import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# YouTube API Configuration
YOUTUBE_API_KEY = "AIzaSyCLrMoPxDQZdI6uLIYfvtFHAWsuwWpe8Y0"
MAX_COMMENTS_PER_VIDEO = 100

# Data Paths
ANNOTATED_DATA_DIR = os.path.join(PROJECT_ROOT, "annotated")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models/saved")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# NLP Configuration
NLTK_RESOURCES = ['punkt', 'stopwords', 'wordnet']
SPACY_MODEL = 'en_core_web_sm'

# Model Configuration
TEST_SIZE = 0.15  # Decreased test size to have more training data
RANDOM_STATE = 42
MAX_FEATURES = 20000  # Increased from 15000
N_ESTIMATORS = 300  # Increased from 250

# Model parameters for SVM
C_VALUE = 15.0  # Increased from 10.0
MAX_ITER = 7500  # Increased from 5000
KERNEL = 'linear'

# Training parameters
VALIDATION_SIZE = 0.1
CLASS_WEIGHT = 'balanced'
CROSS_VALIDATION_FOLDS = 5  # Added cross-validation parameter

# Advanced feature extraction
TEXT_PREPROCESSING = {
    'remove_stopwords': True,
    'stemming': False,
    'lemmatization': True,
    'min_word_length': 2,
    'remove_punctuation': True,
    'lowercase': True,
    'expand_contractions': True,  # New parameter
    'preserve_negation': True     # New parameter
}

# Expanded sentiment lexicons
POSITIVE_WORDS = [
    'good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'best', 'perfect', 'brilliant',
    'fantastic', 'superb', 'wonderful', 'outstanding', 'impressive', 'stellar', 'terrific',
    'solid', 'nice', 'beautiful', 'stunning', 'incredible', 'phenomenal', 'magnificent',
    'flawless', 'exceptional', 'premium', 'top-notch', 'superb', 'delightful', 'marvelous',
    'innovative', 'revolutionary', 'robust', 'reliable', 'durable', 'superior', 'satisfied',
    'responsive', 'seamless', 'smooth', 'crisp', 'vibrant', 'sleek', 'elegant', 'intuitive',
    'powerful', 'fast', 'quick', 'speedy', 'snappy', 'efficient', 'effective', 'improved',
    'upgraded', 'enhanced', 'optimized', 'refined', 'polished', 'professional', 'extraordinary',
    'pleased', 'impressive', 'satisfy', 'satisfies', 'happy', 'joy', 'enjoying', 'enjoy', 'liked',
    'impressed', 'praise', 'praising', 'recommended', 'recommend', 'worth', 'worthy', 'favorite',
    'standout', 'value', 'valuable', 'quality', 'useful', 'satisfied', 'loving', 'loved'
]

NEGATIVE_WORDS = [
    'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst', 'disappointing', 'disappointed',
    'hate', 'dislike', 'mediocre', 'inferior', 'useless', 'ugly', 'fail', 'failure',
    'problem', 'issue', 'defect', 'weak', 'slow', 'glitch', 'buggy', 'expensive', 'overpriced',
    'cheap', 'flimsy', 'fragile', 'unreliable', 'inconsistent', 'frustrating', 'annoying',
    'irritating', 'cumbersome', 'clunky', 'bulky', 'heavy', 'subpar', 'lackluster', 'underwhelming',
    'outdated', 'obsolete', 'sluggish', 'laggy', 'freezes', 'crashes', 'breaks', 'malfunctions',
    'defective', 'faulty', 'error', 'bug', 'broken', 'unresponsive', 'inconvenient', 'awkward',
    'clumsy', 'difficult', 'uncomfortable', 'impractical', 'unusable', 'wasteful', 'regret',
    'dissatisfied', 'complained', 'complaining', 'complaint', 'struggling', 'struggle', 'messed',
    'mess', 'messy', 'pain', 'painful', 'sucks', 'suck', 'garbage', 'trash', 'waste', 'wasted',
    'lacking', 'lacks', 'lack', 'upset', 'avoid', 'avoiding', 'avoided', 'terrible', 'terrible',
    'dissatisfying', 'falling short', 'short', 'shortcoming', 'fell short', 'failure', 'fails'
]

NEUTRAL_WORDS = [
    'okay', 'ok', 'decent', 'average', 'standard', 'normal', 'common', 'alright',
    'fine', 'regular', 'typical', 'usual', 'moderate', 'fair', 'acceptable', 'medium',
    'middle', 'ordinary', 'so-so', 'passable', 'tolerable', 'satisfactory', 'sufficient',
    'adequate', 'neither good nor bad', 'mediocre', 'intermediate'
]

# Intensifiers and Negators
INTENSIFIERS = [
    'very', 'really', 'extremely', 'incredibly', 'absolutely', 'totally',
    'highly', 'completely', 'utterly', 'especially', 'particularly', 'quite',
    'indeed', 'notably', 'decidedly', 'exceedingly', 'remarkably', 'extraordinarily',
    'undoubtedly', 'unquestionably', 'uncommonly', 'unusually', 'terribly',
    'awfully', 'exceptionally', 'super', 'so', 'too', 'most', 'seriously'
]

NEGATORS = [
    'not', 'never', 'no', 'none', 'neither', 'nor', "doesn't", "don't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "can't", "couldn't", "shouldn't", "wouldn't",
    "won't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "hardly",
    "seldom", "rarely", "scarcely", "barely", "nothing", "nobody", "nowhere", "few"
]

# Expanded aspect keywords for more comprehensive detection
ASPECT_KEYWORDS = {
    'camera': [
        'camera', 'photo', 'picture', 'image', 'photography', 'video', 'selfie', 'lens', 'portrait',
        'zoom', 'megapixel', 'pixel', 'sensor', 'aperture', 'stabilization', 'night mode',
        'ultra-wide', 'telephoto', 'macro', 'hdr', 'lowlight', 'low-light', 'recording',
        'resolution', 'shutter', 'capture', 'clarity', 'sharpness', 'focus', 'autofocus',
        'ultrawide', 'cinematic', 'panorama', 'flash', 'depth', 'bokeh', 'blurry', 'grainy',
        'cam', 'shoot', 'shot', 'shooting', 'recorded', 'filming', 'film', 'pics', 'pictures',
        'snapping', 'snapshot', 'exposure', 'photograph', 'photographer', 'photographic',
        'camera quality', 'iso', 'optical', 'digital zoom', 'slow motion', 'timelapse',
        'dynamic range', 'imaging', 'frame rate', 'fps', '4k', '8k', 'ultra hd', 'full hd'
    ],
    'battery': [
        'battery', 'charging', 'power', 'drain', 'life', 'backup', 'fast charge', 'battery life',
        'endurance', 'longevity', 'capacity', 'mah', 'runtime', 'standby', 'watt', 'consumption',
        'discharge', 'energy', 'juice', 'saver', 'efficiency', 'usb-c', 'plug', 'adapter',
        'wireless', 'quick', 'rapid', 'overnight', 'trickle', 'power bank', 'portable', 'outlet',
        'recharge', 'charge cycle', 'low battery', 'percentage', 'dies', 'lasts', 'dead',
        'power drain', 'power hungry', 'charger', 'charging time', 'charging speed', 'full charge',
        'charge time', 'battery capacity', 'battery percentage', 'battery indicator', 'battery saving',
        'battery saver', 'power saving', 'battery optimization', 'power mode', 'lightning',
        'volt', 'ampere', 'amp', 'milliamp', 'milliampere', 'type-c', 'power delivery'
    ],
    'performance': [
        'performance', 'speed', 'fast', 'slow', 'lag', 'processing', 'snappy', 'processor',
        'benchmark', 'fps', 'frame rate', 'cpu', 'gpu', 'chip', 'chipset', 'ram', 'memory',
        'storage', 'responsiveness', 'multitasking', 'gaming', 'throttle', 'heat', 'overheating',
        'loading', 'boot', 'restart', 'crash', 'freeze', 'stutter', 'optimization', 'efficient',
        'smooth', 'quick', 'seamless', 'fluent', 'snappy', 'zippy', 'responsive', 'powerful',
        'robust', 'capable', 'swift', 'rapid', 'nimble', 'agile', 'clunky', 'sluggish',
        'faster', 'slower', 'hanging', 'frozen', 'reboot', 'app switching', 'app launch',
        'animation', 'transitions', 'gigabyte', 'gb', 'tb', 'terabyte', 'megabyte', 'mb',
        'clock speed', 'ghz', 'gigahertz', 'benchmark', 'antutu', 'geekbench', '3dmark',
        'core', 'thermal', 'heating', 'temperature', 'soc', 'system on chip'
    ],
    'display': [
        'display', 'screen', 'resolution', 'bright', 'color', 'refresh', 'hdr', 'contrast',
        'oled', 'lcd', 'amoled', 'ips', 'panel', 'refresh rate', 'nits', 'brightness',
        'dimming', 'vibrant', 'saturation', 'calibration', 'accurate', 'viewing angle',
        'bezels', 'notch', 'hole-punch', 'ppi', 'pixel density', 'qhd', 'fhd', 'hd', '4k',
        'dynamic', 'adaptive', 'touch', 'responsive', 'scrolling', 'smooth', 'animation',
        'vivid', 'text', 'readability', 'sunlight', 'outdoor', 'visibility', 'eye strain',
        'reflective', 'glare', 'colors', 'colorful', 'whites', 'blacks', 'deep blacks',
        'washed out', 'screen size', 'inch', 'diagonal', 'aspect ratio', 'curved',
        'flat', 'edge', 'edge-to-edge', 'bezel-less', 'full-screen', 'punch hole',
        'water drop', 'tear drop', 'color gamut', 'color accuracy', 'color reproduction',
        'hdr10', 'hdr10+', 'dolby vision', 'always on', 'always-on'
    ],
    'design': [
        'design', 'build', 'feel', 'body', 'glass', 'metal', 'plastic', 'premium', 'look',
        'weight', 'grip', 'size', 'slim', 'thick', 'aesthetic', 'ergonomic', 'comfort',
        'sturdy', 'construction', 'durable', 'material', 'texture', 'finish', 'color', 'hue',
        'fingerprint', 'smudge', 'scratch', 'resistant', 'rugged', 'protection', 'waterproof',
        'water-resistant', 'ip68', 'ip67', 'dust', 'gorilla glass', 'ceramic', 'stainless steel',
        'aluminum', 'lightweight', 'compact', 'bulky', 'dimensions', 'curves', 'flat', 'edges',
        'buttons', 'ports', 'symmetry', 'unibody', 'craftsmanship', 'build quality',
        'hand feel', 'in-hand feel', 'hand fit', 'comfortable', 'uncomfortable', 'slippery',
        'width', 'height', 'thickness', 'thin', 'sleek', 'symmetrical', 'asymmetrical',
        'usb port', 'headphone jack', 'sim tray', 'speaker grille', 'microphone', 'volume rocker',
        'power button', 'button layout', 'button feel', 'clicky', 'mushy', 'tactile'
    ],
    'price': [
        'price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'affordable', 'overpriced',
        'budget', 'flagship', 'money', 'premium', 'pricey', 'economical', 'inexpensive',
        'bargain', 'deal', 'discount', 'sale', 'investment', 'dollars', 'euros', 'rupees',
        'yuan', 'yen', 'pounds', 'mid-range', 'high-end', 'entry-level', 'tier', 'segment',
        'competitive', 'justified', 'reasonable', 'unreasonable', 'pay', 'spend', 'purchase',
        'costly', 'extravagant', 'economize', 'compromise', 'trade-off', 'sacrifice', 'splurge',
        'save', 'wasteful', 'bang for buck', 'value proposition', 'pricing',
        'value for money', 'costly', 'costs', 'affordable', 'unaffordable', 'low-cost',
        'high-cost', 'budget-friendly', 'budget friendly', 'price tag', 'price point',
        'priced', 'premium price', 'premium-priced', 'competitively priced', 'worth the money',
        'not worth', 'price-to-performance', 'bang-for-your-buck', 'spending', 'spent'
    ]
}

# Model ensembling configuration
ENSEMBLE_WEIGHTS = {
    'svm': 0.7,
    'rule_based': 0.3
}

# Aspect sentiment threshold configuration
ASPECT_CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence to trust model prediction

# Data augmentation settings
DATA_AUGMENTATION = {
    'enabled': True,
    'synonym_replacement': True,
    'random_swap': True,
    'random_deletion': False
}
