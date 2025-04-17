import re
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from distilbert import clean_text_for_distilbert, analyze_sentiment_bert, extract_aspect_sentiment, PREDEFINED_ASPECTS

def get_valid_keyword():
    keyword = st.text_input("Enter keyword to search for:", key="keyword_input").strip()
    if keyword and re.search(r"[a-zA-Z0-9]", keyword):
        return keyword
    elif keyword != "":
        st.error("❌ Keyword cannot be empty or contain only special characters.")
    return None


def get_valid_subreddit(reddit):
    subreddit = st.text_input("Enter subreddit (leave blank for all):", key="subreddit_input").strip()
    if not subreddit:
        return None  # No subreddit = search all

    try:
        reddit.subreddits.search_by_name(subreddit, exact=True)
        return subreddit
    except Exception:
        st.error(f"❌ The subreddit '{subreddit}' does not exist or could not be verified.")
        return None
    
# Function to get sentence embeddings
def get_embedding(embedding_model, text):
    if isinstance(text, str):  # Ensure the input is a string
        return embedding_model.encode(text)
    else:
        raise ValueError("Input must be a string.")


# Function to calculate cosine similarity
def calculate_similarity(embedding_model, text1, text2):
    embedding1 = get_embedding(embedding_model, text1)
    embedding2 = get_embedding(embedding_model, text2)
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

def load_sentence_transformer():
    """Load and return the SentenceTransformer model with error handling."""
    try:
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("SentenceTransformer loaded successfully within the function.")
        return embedding_model
    except Exception as e:
        error_message = f"Error loading SentenceTransformer within the function: {e}"
        print(error_message)
        st.error(f"❌ {error_message}")  # Display error in Streamlit
        return None

def fetch_comments_with_semantic_filtering(reddit, keyword, embedding_model, subreddit=None, sorting='new', similarity_threshold=0.5):
    """Fetches comments from Reddit, filters based on semantic similarity with the keyword, and returns cleaned comment data.

    Args:
        reddit: An authenticated PRAW Reddit instance.
        keyword: The keyword to search for.
        embedding_model: The SentenceTransformer model for semantic similarity.
        subreddit: The name of the subreddit (optional). If None, searches across all of Reddit.
        sorting: The sorting method for posts ('new', 'hot', 'top', 'relevance'). Defaults to 'new'.
        similarity_threshold: The minimum semantic similarity score for a comment to be included.

    Returns:
        pd.DataFrame: DataFrame with filtered and cleaned comments and their timestamps.
    """
    POST_LIMIT = 500  # Number of posts to fetch
    COMMENT_LIMIT = 20  # Max comments per post

    if not keyword.strip():
        print("Error: Keyword cannot be empty. Please enter a valid search term.")
        return pd.DataFrame()

    comments_data = []

    try:
        # Handle subreddit or all subreddits
        if subreddit:
            sub = reddit.subreddit(subreddit)  # Correct way to access subreddit
        else:
            sub = reddit.subreddit('all')  # Use 'all' if no subreddit is specified

        print(f"Searching in subreddit: {sub.display_name if subreddit else 'All subreddits'}")

        # Search for posts with keyword
        if sorting == 'new':
            search = sub.search(f'"{keyword}"', sort="new", limit=POST_LIMIT)
        elif sorting == 'hot':
            search = sub.search(f'"{keyword}"', sort="hot", limit=POST_LIMIT)
        elif sorting == 'top':
            search = sub.search(f'"{keyword}"', sort="top", limit=POST_LIMIT)
        else:  # Default to relevance
            search = sub.search(f'"{keyword}"', sort="relevance", limit=POST_LIMIT)

        posts_found = 0  # Track if any posts are found

        for post in search:
            posts_found += 1
            post.comments.replace_more(limit=0)  # Only fetch top-level comments
            top_comments = post.comments.list()[:COMMENT_LIMIT]  # Limit comments per post

            for comment in top_comments:
                similarity = calculate_similarity(embedding_model, comment.body, keyword)

                if similarity > similarity_threshold:
                    cleaned_text = clean_text_for_distilbert(comment.body)

                    if cleaned_text:
                        # Add relevant comment info (timestamp, cleaned text)
                        comments_data.append({
                            "Timestamp": pd.to_datetime(comment.created_utc, unit='s'),
                            "Cleaned Comment": cleaned_text
                        })

        if posts_found == 0:
            print("No posts found for the given keyword in this subreddit.")
        else:
            print(f"Found {posts_found} posts related to '{keyword}'.")

    except Exception as e:
        print(f"Error fetching comments: {e}")
        return pd.DataFrame()

    # Return DataFrame with filtered comments
    df_comments = pd.DataFrame(comments_data)

    return df_comments

def fetch_and_analyze_sentiment(reddit, tokenizer, model, embedding_model, keyword, subreddit=None, sorting='new', similarity_threshold=0.5):
    """Fetches and analyzes sentiment using both semantic filtering and aspect-based sentiment analysis."""

    print(f"Fetching comments with semantic filtering (sorted by '{sorting}')...")
    df_comments = fetch_comments_with_semantic_filtering(reddit, keyword, embedding_model, subreddit, sorting, similarity_threshold)
    print("✅ Done fetching comments")

    if df_comments.empty:
        print("❌ No relevant comments found after filtering.")
        return pd.DataFrame()  # Return empty DataFrame if no comments

    # Analyze sentiment for each cleaned comment using the BERT model
    sentiment_data = []

    # Define the columns for aspect sentiment based on your predefined aspects
    predefined_aspects = list(PREDEFINED_ASPECTS.keys())  # Get the aspect names
    print("✅ Predefined aspects retrieved")

    for _, row in df_comments.iterrows():
        cleaned_comment = row["Cleaned Comment"]

        # Analyze sentiment using the DistilBERT model
        print(f"Analyzing sentiment for comment: {cleaned_comment[:50]}...")  # Print a preview
        sentiment_score = analyze_sentiment_bert(tokenizer, model, cleaned_comment)

        # Ensure sentiment_score is a scalar value (extract from NumPy array if needed)
        if isinstance(sentiment_score, list):
            sentiment_score = sentiment_score[0]  # Extract from list (as you had)
        elif isinstance(sentiment_score, np.ndarray):
            sentiment_score = sentiment_score.item()  # Extract the scalar from NumPy array

        print(f"✅ Sentiment score calculated: {sentiment_score}")

        # Extract aspect-based sentiment for the comment
        aspect_sentiment = extract_aspect_sentiment(tokenizer, model, cleaned_comment)
        print(f"✅ Aspect sentiment extracted: {aspect_sentiment}")

        # Add the aspect sentiments into individual columns
        aspect_sentiments_row = {aspect: aspect_sentiment.get(aspect, None) for aspect in predefined_aspects}
        print("✅ Aspect sentiments mapped")

        sentiment_data.append({
            "Timestamp": row["Timestamp"],
            "Cleaned Comment": cleaned_comment,
            "Sentiment Score": sentiment_score,
            **aspect_sentiments_row  # Add aspect sentiment columns dynamically
        })
        print("✅ Comment processing complete")

    # Return a DataFrame with sentiment analysis results, including separate columns for each aspect's sentiment
    df_sentiment = pd.DataFrame(sentiment_data)
    print("✅ DataFrame created successfully")

    return df_sentiment