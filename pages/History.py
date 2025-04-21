import streamlit as st
from auth import get_user_history 
import pandas as pd
from report import get_colour, map_sentiment_to_label
import plotly.graph_objects as go

st.title("Your Analysis History")

if 'logged_in' in st.session_state and st.session_state['logged_in'] and 'user' in st.session_state and st.session_state['user']:
    user_id = st.session_state['user']
    history_data, error = get_user_history(user_id)

    if error:
        st.error(f"Error loading history: {error}")
    elif history_data:
        st.subheader("Past Analyses:")
        for item in history_data:
            timestamp_str = item['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if 'timestamp' in item else 'N/A'
            keyword_str = item.get('keyword', 'N/A')
            subreddit_str = item.get('subreddit', 'All')
            sorting_str = item.get('sorting', 'N/A')

            st.markdown(f"**Timestamp:** {timestamp_str}  \n**Keyword:** `{keyword_str}`  \n**Subreddit:** `{subreddit_str}`  \n**Sorting:** `{sorting_str}`")

            if 'data' in item:
                df_history_item = pd.DataFrame(item['data'])

                # General sentiment
                if "Sentiment Score" in df_history_item.columns:
                    avg_sentiment = df_history_item["Sentiment Score"].mean()
                    sentiment_label = map_sentiment_to_label(avg_sentiment)
                    sentiment_color = get_colour(avg_sentiment)
                    st.markdown(
                        f"<h4 style='color:{sentiment_color};'>Average Reddit User Sentiment: {sentiment_label} ({avg_sentiment:.2f})</h4>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Sentiment Score column not found in the analysis data.")

                # Aspect-based sentiment
                ignored_cols = ['timestamp', 'Sentiment Score', 'Cleaned Comment', 'Comment', 'Sentiment Label']
                aspect_cols = [col for col in df_history_item.columns if col not in ignored_cols and df_history_item[col].dtype in ['float64', 'int64']]

                if aspect_cols:
                    aspect_avg = df_history_item[aspect_cols].mean().reset_index()
                    aspect_avg.columns = ['Aspect', 'Average Score']
                    aspect_avg['Sentiment Label'] = aspect_avg['Average Score'].apply(map_sentiment_to_label)
                    aspect_avg['Color'] = aspect_avg['Average Score'].apply(get_colour)

                    # Radar chart
                    categories = aspect_avg['Aspect'].tolist()
                    values = aspect_avg['Average Score'].tolist()

                    radar_fig = go.Figure()
                    radar_fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Average Sentiment',
                        line_color='indigo'
                    ))
                    radar_fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[-1, 1])
                        ),
                        showlegend=False,
                        title="Aspect-Based Sentiment Radar"
                    )
                    st.plotly_chart(radar_fig, use_container_width=True)

                    # Expandable: View aspect sentiment table
                    with st.expander("View Aspect Score Table"):
                        st.dataframe(aspect_avg.drop(columns=['Color']))
                else:
                    st.info("No aspect-related sentiment data found in this analysis.")

                # Full table in expandable
                with st.expander("View Full Table"):
                    st.dataframe(df_history_item)

            st.markdown("---")
    else:
        st.info("You haven't performed any analyses yet.")
else:
    st.info("Please log in to view your analysis history.")
