import pandas as pd
import tweepy
from openai import OpenAI
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize APIs
client = OpenAI(api_key=os.getenv('***'))
twitter_client = tweepy.Client(
    bearer_token=os.getenv('***'),
    consumer_key=os.getenv('***'),
    consumer_secret=os.getenv('***'),
    access_token=os.getenv('***'),
    access_token_secret=os.getenv('***')
)

def get_tweet_comments(article_title):
    try:
        # Search for NYT tweet containing article title
        query = f'from:nytimes "{article_title}"'
        tweets = twitter_client.search_recent_tweets(query=query, max_results=10)

        if not tweets.data:
            return None

        # Get tweet ID and fetch replies
        tweet_id = tweets.data[0].id
        replies = twitter_client.search_recent_tweets(
            query=f'conversation_id:{tweet_id}',
            max_results=100
        )

        return [reply.text for reply in replies.data] if replies.data else None

    except Exception as e:
        print(f"Error fetching tweets for {article_title}: {e}")
        return None

def get_sentiment_score(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You analyze text sentiment."},
                {"role": "user", "content": f"Without considering the topic but
                 only with the words and the tone of writing give me a positivity
                  score between 0 and 1 with 0 meaning the sentence is completely
                 negative and 1 completely positive. Only return the number.\n\nText: {text}"}
            ]
        )
        return float(response.choices[0].message.content.strip())
    except:
        return None

# Load articles
articles_path = '/home/henrijn/Documents/SHS/Iran/data/filtered_nytimes.csv'
nyt_articles = pd.read_csv(articles_path)

# Extract page numbers and dates
pages = nytimes.iloc[:, 7]
dates = nytimes.iloc[:, 3]

# Calculate weights using formula
weights = np.exp(dates - 2010) / np.log(pages)

# Process each article
sentiment_scores = []
for title in nyt_articles.iloc[:, 1]:  # Column 1 contains titles
    comments = get_tweet_comments(title)
    if comments:
        # Get sentiment for each comment
        scores = [get_sentiment_score(comment) for comment in comments]
        scores = [s for s in scores if s is not None]
        avg_score = sum(weights*scores) / len(scores) if scores else None
        sentiment_scores.append(avg_score)
    else:
        sentiment_scores.append(None)
    time.sleep(1)  # Rate limiting

# Add scores to DataFrame
nyt_articles['sentiment_score'] = sentiment_scores

# Save results
output_path = '/home/henrijn/Documents/SHS/Iran/data/filtered_nytimes_with_sentiment.csv'
nyt_articles.to_csv(output_path, index=False)
