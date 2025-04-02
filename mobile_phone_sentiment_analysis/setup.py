"""
Setup script for the mobile phone sentiment analysis package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="mobile_phone_sentiment_analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A specialized sentiment analysis system for mobile phone reviews from YouTube comments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mobile_phone_sentiment_analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "phone_sentiment=mobile_phone_sentiment_analysis.main:main",
        ],
    },
) 