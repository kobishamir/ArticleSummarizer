# Article Summarizer

## Description
This Python script fetches articles from provided URLs and uses natural language processing to summarize them. The script utilizes the `requests` library for fetching the article, `BeautifulSoup` for parsing the HTML, and Hugging Face's `transformers` pipeline with the `facebook/bart-large-cnn` model for summarization.
Read about [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn).


## Features
- Fetches articles from any URL.
- Summarizes the content using advanced NLP models.
- TODO: Implement handling for long articles.
- TODO: Enable GPU acceleration for improved performance.

## Installation
Clone this repository and install the required libraries:
```bash
git clone https://github.com/yourusername/article-summarizer.git
cd article-summarizer
pip install -r requirements.txt
```

## Usage
Run the script and follow the prompt to enter the URL of the article you wish to summarize:
```bash
python summarize.py
```

## Requirements
- Python 3.x
- requests
- bs4 (BeautifulSoup)
- transformers
