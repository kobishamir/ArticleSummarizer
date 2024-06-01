import requests
from bs4 import BeautifulSoup
import torch
from transformers import pipeline, AutoTokenizer
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def fetch_article(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        return article_text
    else:
        logger.error(f"Failed to fetch article, status code: {response.status_code}")
        return None


def split_text(text, tokenizer, max_length=1024):
    tokens = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start = end

    return chunks


def summarize_article(article_text):
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise use CPU
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    chunks = split_text(article_text, tokenizer)

    for i, chunk in enumerate(chunks):
        token_count = len(tokenizer(chunk, return_tensors='pt')['input_ids'][0])
        logger.debug(f"Chunk {i + 1}/{len(chunks)}: {token_count} tokens")
        if token_count > 1024:
            logger.warning(f"Chunk {i + 1} exceeds max token length: {token_count} tokens")
        logger.debug(chunk[:500])  # Log the first 500 characters of the chunk for inspection

    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Error summarizing chunk {i + 1}: {e}")
            continue

    return ' '.join(summaries)


if __name__ == "__main__":
    url = input("Enter the URL of the article: ")
    article = fetch_article(url)
    if article:
        logger.info("Article fetched successfully!")
        summary = summarize_article(article)
        print("\nSummary:")
        print(summary)
    else:
        logger.error("Failed to fetch the article.")