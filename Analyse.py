import streamlit as st
st.set_page_config(initial_sidebar_state="collapsed")
from PIL import Image
import time
import re
import pandas as pd
import prawcore
import matplotlib.pyplot as plt
from auth import authenticate_reddit, register_user, login_user, logout_user, save_user_history
from distilbert import load_distilbert
from getcomments import load_sentence_transformer, fetch_and_analyze_sentiment, PREDEFINED_ASPECTS
from report import get_colour, plot_aspect_radar_chart, map_sentiment_to_label, display_sentiment_distribution,display_aspect_contribution_to_sentiment, TECH_CATEGORIES

def main():
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'reddit' not in st.session_state:
        st.session_state['reddit'] = None
    if 'tokenizer' not in st.session_state:
        st.session_state['tokenizer'] = None
    if 'model' not in st.session_state:
        st.session_state['model'] = None
    if 'embedding_model' not in st.session_state:
        st.session_state['embedding_model'] = None
    if 'df_comments' not in st.session_state:
        st.session_state['df_comments'] = pd.DataFrame()

    if not st.session_state['logged_in']:
        # Set sidebar state to collapsed when not logged in
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] {
                    display: none;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.title("Welcome to the Specialised Reddit Sentiment Analyzer for Tech Products")
        choice = st.radio("Login or Register?", ["Login", "Register"])

        if choice == "Login":
            st.subheader("Login")
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type='password')
            if st.button("Login"):
                success, uid, error = login_user(login_username, login_password)
                if success:
                    st.session_state['user'] = uid
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = login_username
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error(f"Login failed: {error}")

        elif choice == "Register":
            st.subheader("Register")
            new_username = st.text_input("Username")
            new_email = st.text_input("Email Address")
            new_password = st.text_input("Password", type='password')
            if st.button("Register"):
                success, error = register_user(new_username, new_email, new_password)
                if success:
                    st.success("Registration successful! Please log in with your username and password.")
                else:
                    st.error(f"Registration failed: {error}")
    else:
        # Re-enable the sidebar (it will be visible by default after this point)
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] {
                    display: block; /* Or leave it to Streamlit's default */
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        image_path = "images/sentiments.jpg"
        st.image(Image.open(image_path), use_container_width=True)
        
        st.title("Specialised Reddit Sentiment Analyzer for Tech Products")

        st.sidebar.subheader(f"Current User: {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            logout_user()
            st.success("Logged out successfully!")
            st.session_state['reddit'] = None
            st.session_state['tokenizer'] = None
            st.session_state['model'] = None
            st.session_state['embedding_model'] = None
            st.session_state['df_comments'] = pd.DataFrame() # Clear previous results
            st.rerun()

        with st.expander("ℹ️ Info and Instructions"):
            st.info("""
            **ℹ️ About this Tool:**

            This application is designed to perform sentiment analysis on Reddit discussions related to tech products.
            It allows you to search for specific keywords across Reddit or within particular subreddits and
            analyze the overall sentiment and the sentiment towards different aspects of the topic.

            **💬 How Reddit Works (Brief Overview):**

            Reddit is a social news aggregation and discussion website where users can share content,
            links, and text posts. Content is organized into communities called **subreddits**.

            * **Subreddits:** These are focused communities dedicated to specific topics (e.g., r/technology, r/gadgets, r/apple).
                When you enter a subreddit in the tool, the analysis will be limited to discussions within that specific community.
                Leaving the subreddit field blank will search across all public subreddits on Reddit.

            * **Posts:** Users submit posts, which can be text-based or links to external content.

            * **Comments:** Other users can comment on these posts, creating discussions and expressing their opinions. This tool analyzes the sentiment expressed in these comments.

            * **Sorting:** Reddit allows users to sort posts in various ways, such as:
                * **🔥 Hot:** Currently popular discussions.
                * **🆕 New:** The most recently submitted posts.
                * **🔝 Top:** Posts with the highest number of upvotes (can be filtered by time).
                * **Relevance:** Results that best match your search query (default).
                You can select your preferred sorting method below to focus your analysis.

            **⚙️ How to Use:**

            1.  Enter the **keyword** you want to analyze (e.g., "iPhone 15", "Samsung Galaxy S23") and be specific!.
            2.  Optionally, enter a specific **subreddit** to focus your analysis on (e.g., "apple", "gaming"). Leave blank to search all of Reddit.
            3.  Select how you want to **sort** the Reddit posts before analyzing their comments (New, Hot, Top, most relevant).
            4.  Click the **"Run Analysis"** button to start the process.

            The tool will then fetch relevant comments, perform sentiment analysis, and display a visualization of the sentiment towards different aspects of your chosen keyword.
            """)


        # Show tech categories
        with st.container():
            st.subheader("Popular Tech Subreddits")
            df_categories = pd.DataFrame([(cat, ", ".join(subs)) for cat, subs in TECH_CATEGORIES.items()],
                                         columns=["Category", "Popular Subreddits"])
            st.dataframe(
                df_categories,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Category": st.column_config.Column(width="small"),
                    "Popular Subreddits": st.column_config.Column(width=None) # Set width to None for auto-sizing
                }
            )

        st.markdown("---")

        # Input fields
        keyword = st.text_input("🔍 Enter keyword to search for:").strip()
        subreddit = st.text_input("📌 Enter subreddit (leave blank for all):").strip()
        sorting_options = ["new", "hot", "top", "most relevant"]
        sorting = st.selectbox("Sort posts by:", sorting_options)

        # Run button
        if st.button("Run Analysis"):
            # Keyword validation
            if not keyword:
                st.error("❌ Keyword cannot be empty.")
                return
            if not re.search(r"[a-zA-Z0-9]", keyword):
                st.error("❌ Keyword must contain valid alphanumeric characters.")
                return

            # Ensure Reddit is authenticated
            if st.session_state.get('reddit') is None:
                st.session_state['reddit'] = authenticate_reddit()
                if st.session_state['reddit'] is None:
                    st.error("Reddit connection failed.")
                    return
            reddit = st.session_state['reddit'] # Assign the authenticated instance to the local 'reddit' variable

            # Subreddit validation
            if subreddit:
                try:
                    reddit = st.session_state['reddit'] # Use the authenticated instance
                    if not reddit:
                        st.error("Reddit connection failed.")
                        return
                    reddit.subreddits.search_by_name(subreddit, exact=True)
                except prawcore.exceptions.NotFound:
                    st.error(f"❌ Subreddit '{subreddit}' does not exist.")
                    return
                except Exception as e:
                    st.error(f"An error occurred while checking the subreddit: {e}")
                    return
                else:
                    subreddit = subreddit  # Keep the valid subreddit
            else:
                subreddit = None  # Accept blank input

            # Load models if not already loaded
            if st.session_state.get('tokenizer') is None or st.session_state.get('model') is None or st.session_state.get('embedding_model') is None:
                with st.spinner("Loading models..."):
                    try:
                        st.session_state['tokenizer'], st.session_state['model'] = load_distilbert()
                        st.session_state['embedding_model'] = load_sentence_transformer()
                        #st.success("✅ Models loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to load models: {e}")
                        return

            # Fetch and analyze with spinner
            with st.spinner(text=f"Gathering and analysing Reddit comments for '{keyword}' (sorted by {sorting})..."):
                df_comments = fetch_and_analyze_sentiment(
                    reddit,
                    st.session_state['tokenizer'],
                    st.session_state['model'],
                    st.session_state['embedding_model'],
                    keyword=keyword,
                    subreddit=subreddit,
                    sorting=sorting
                )

                if not df_comments.empty:
                    st.session_state['df_comments'] = df_comments
                    st.success("Analysis complete! Scroll down to see the results.") # Provide feedback
                else:
                    st.warning("⚠️ No comments found for the given keyword and subreddit.")
        else:
            st.info("Enter a keyword and click 'Run Analysis' to begin.")

        # Conditionally display the sentiment report and visualizations
        if not st.session_state['df_comments'].empty:
            st.subheader("📄 Sentiment Report")

            # Display the resulting DataFrame
            st.subheader("Filtered and analyzed Reddit comments")
            with st.expander("🔍 View Filtered Comments Breakdown"):
                st.dataframe(st.session_state['df_comments'])

            #CSS for containers
            st.markdown("""
                <style>
                    .graph-container {
                        border: 2px solid grey;
                        padding: 20px;
                        border-radius: 10px;
                    }
                </style>
            """, unsafe_allow_html=True)

            average_sentiment = st.session_state['df_comments']["Sentiment Score"].mean()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class="graph-container">
                    <h5>Average Sentiment Score</h5>
                    <h2>{average_sentiment:.2f}</h2>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                sentiment_color = get_colour(average_sentiment)
                st.markdown(f"""
                    <div class="graph-container" style="background-color: {sentiment_color};">
                    <h5>Sentiment Interpretation</h5>
                    <h2>{map_sentiment_to_label(average_sentiment)}</h2>
                    </div>
                """, unsafe_allow_html=True)

            st.divider()
            col3, col4 = st.columns(2)
            with col3:
                display_sentiment_distribution(st.session_state['df_comments'])
                st.markdown("""
                **Distribution of Overall Sentiment:** This bar chart visually represents the distribution of sentiment expressed in the analyzed Reddit comments.
                """)
            with col4:
                st.subheader("Aspect Contribution to Sentiment") # Add a subheader here
                display_aspect_contribution_to_sentiment(PREDEFINED_ASPECTS.keys())

            st.divider()
            plot_aspect_radar_chart(st.session_state['df_comments'], keyword, subreddit)
            st.markdown("""
            Aspect Sentiment Radar: This chart visualizes the average sentiment (0.0-1.0) for key aspects (e.g., Features, Performance) discussed on reddit for the analysed product. Points further from the center indicate more positive sentiment towards that aspect, with line color providing a qualitative sentiment indication (see legend).
            """)
            # Save history (remains the same)
            if st.session_state['user']:
                success, error = save_user_history(
                    st.session_state['user'],
                    st.session_state['df_comments'],
                    keyword,
                    sorting,
                    subreddit
                )
                if success:
                    st.success("Analysis results saved to your history.")
                else:
                    st.error(f"Error saving history: {error}")
            else:
                st.info("Please log in to save your analysis history.")

if __name__ == "__main__":
    main()