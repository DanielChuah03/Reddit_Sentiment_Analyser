import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
from firebase_admin import firestore
from dotenv import load_dotenv
import praw
import streamlit as st

# --- Initialize Firebase Admin SDK ---
try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate("credentials/firebase_key.json")  # Replace with your key path
    firebase_admin.initialize_app(cred)



db = firestore.client()

def register_user(username, email, password):
    if not username.strip():
        return False, "Username cannot be empty."
    if len(username.strip()) <= 3:
        return False, "Username must be longer than 3 characters."
    if len(username.strip()) > 50:  # Example maximum length for username
        return False, "Username cannot be longer than 50 characters."
    if not password.strip():
        return False, "Password cannot be empty."
    if len(password) <= 3:
        return False, "Password must be longer than 3 characters."
    if len(password) > 100:  # Example maximum length for password
        return False, "Password cannot be longer than 100 characters."
    if not email.strip():
        return False, "Please enter your email address."
    if len(email.strip()) > 100:  # Example maximum length for email
        return False, "Email address cannot be longer than 100 characters."

    try:
        users_ref = db.collection('users')
        username_query = users_ref.where('username', '==', username).limit(1).get()
        if username_query:
            return False, "Username already exists."

        try:
            user = auth.create_user(
                email=email,
                password=password,
                display_name=username
            )
            # Create a user document in Firestore with username and email
            user_ref = db.collection('users').document(user.uid)
            user_ref.set({'username': username, 'email': email})
            return True, None
        except Exception as e:
            error_message = str(e)
            if "EMAIL_EXISTS" in error_message or "auth/email-already-exists" in error_message:
                return False, "Email address is already in use."
            elif "INVALID_EMAIL" in error_message or "auth/invalid-email" in error_message or "Malformed email address" in error_message:
                return False, "Please enter a valid email address."
            else:
                return False, f"An error occurred during registration: {e}"

    except Exception as e:
        return False, f"An error occurred during registration (outer): {e}"


def login_user(username, password):
    if not username.strip():
        return False, None, "Username cannot be empty."
    if not password.strip():
        return False, None, "Password cannot be empty."

    try:
        users_ref = db.collection('users')
        username_query = users_ref.where('username', '==', username).limit(1).get()

        if username_query:
            user_doc = username_query[0]
            user_data = user_doc.to_dict()
            user_id = user_doc.id

            return True, user_id, None
        else:
            return False, None, "Invalid username or password."
    except Exception as e:
        return False, None, f"An error occurred during login: {e}"
    

def get_user_info(uid):
    try:
        user = auth.get_user(uid)
        return user, None
    except auth.UserNotFoundError:
        return None, "User not found."
    except Exception as e:
        return None, f"An error occurred: {e}"

def save_user_history(uid, df_data, keyword, sorting, subreddit):
    try:
        history_collection = db.collection('users').document(uid).collection('history')
        timestamp_now = firestore.SERVER_TIMESTAMP  # Use server timestamp
        history_collection.add({
            'timestamp': timestamp_now,
            'keyword': keyword,
            'sorting': sorting,
            'subreddit': subreddit if subreddit else "all",  # Save subreddit or "all"
            'data': df_data.to_dict(orient='records')
        })
        return True, None
    except Exception as e:
        return False, f"Error saving history: {e}"
    
def get_user_history(uid):
    try:
        history_collection = db.collection('users').document(uid).collection('history').order_by('timestamp', direction=firestore.Query.DESCENDING)
        history = history_collection.get()
        history_data = [doc.to_dict() for doc in history]
        return history_data, None
    except Exception as e:
        return None, f"Error fetching history: {e}"

def logout_user():
    st.session_state['user'] = None
    st.session_state['logged_in'] = False
    return True

def authenticate_reddit():
    """Loads environment variables and attempts to connect to the Reddit API.

    Returns:
        praw.Reddit or None: A PRAW Reddit instance if initialization is successful,
                              None otherwise.
    """
    load_dotenv()

    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    user_agent = os.getenv("USER_AGENT")

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        return reddit
    except Exception:
        return None