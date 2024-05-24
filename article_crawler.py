import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# TODO: split and pipline the article text for long articles handling
# TODO: try to figure out how to use GPU using cuda for better performance


def fetch_article(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        return article_text
    else:
        return None


def summarize_article(article_text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(article_text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']


if __name__ == "__main__":
    link = input("Enter the URL of the article: ")
    article = fetch_article(link)
    if article:
        print("Article fetched successfully!")
        print(article[:500] + "...")  # Print first 500 characters as a sample

        summarization = summarize_article(article)
        print("\nSummary:")
        print(summarization)
    else:
        print("Failed to fetch the article.")
