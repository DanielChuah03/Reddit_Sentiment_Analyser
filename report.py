import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.patches as mpatches


TECH_CATEGORIES = {
    "Smartphones": [
        "r/Android", "r/apple", "r/smartphones",
    ],
    "Laptops": [
        "r/laptops", "r/suggestalaptop", "r/windows",
    ],
    "PC Components": [
        "r/buildapc", "r/pcmasterrace", "r/monitors",
    ],
    "Audio": [
        "r/headphones", "r/audiophile", "r/headphoneadvice", "r/earbuds", 
    ],
    "Gaming Consoles": [
        "r/PS5", "r/XboxSeriesX", "r/NintendoSwitch", "r/playstation", 
    ],
    "Wearables": [
        "r/smartwatch", "r/applewatch", "r/WearOS", "r/fitbit",
    ],
    "Cameras & Drones": [
        "r/photography", "r/drones", "r/canon", "r/nikon", "r/videography", 
    ],
    "Tablets & E-readers": [
        "r/tablets", "r/ipad", "r/kindle", "r/androidtablets",
    ],
    "Smart Home Devices": [
        "r/smarthome", "r/homeassistant", "r/amazonecho", "r/googlehome",
    ],
    "VR & AR Headsets": [
        "r/virtualreality", "r/oculus", "r/augmentedreality", 
    ],
}

def map_sentiment_to_label(score):
    if score < 0.2:
        return "Very Negative"
    elif score < 0.4:
        return "Negative"
    elif score < 0.6:
        return "Neutral"
    elif score < 0.8:
        return "Positive"
    else:
        return "Very Positive"

def get_colour(value):
    """Returns a distinct color based on the sentiment value (0 to 1)."""
    if value >= 0.8:
        return 'forestgreen'
    elif value >= 0.6:
        return 'limegreen'
    elif value >= 0.4:
        return 'gold'
    elif value >= 0.2:
        return 'salmon'  
    else:
        return 'firebrick' 
    
def display_sentiment_distribution(df_sentiment):
    # Make a safe copy 
    df = df_sentiment.copy()

    # Create a new column with sentiment labels
    df['Sentiment Label'] = df['Sentiment Score'].apply(map_sentiment_to_label)

    # Define the desired order of sentiment labels
    sentiment_order = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

    # Convert 'Sentiment Label' to categorical with the specified order
    df['Sentiment Label'] = pd.Categorical(df['Sentiment Label'], categories=sentiment_order, ordered=True)

    # Count the labels (the order will now be based on the categorical type)
    sentiment_counts = df['Sentiment Label'].value_counts().sort_index()

    # Plot
    fig, ax = plt.subplots()
    colors = ['red', 'salmon', 'lightgray', 'limegreen', 'forestgreen']
    sentiment_counts.plot(kind='bar', ax=ax, color=colors)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Comments")
    ax.set_title("Distribution of Overall Sentiment")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(fig)

def plot_aspect_radar_chart(df, keyword, subreddit=None):
    # Select columns that are aspect sentiment scores 
    aspect_columns = df.columns[3:]

    valid_aspects = []
    aspect_values = {}
    for aspect in aspect_columns:
        # Count the number of non-NaN values in the aspect column
        non_nan_count = df[aspect].count()
        if non_nan_count >= 3:
            aspect_mean = df[aspect].mean(skipna=True)
            if np.isfinite(aspect_mean):
                valid_aspects.append(aspect)
                aspect_values[aspect] = aspect_mean

    if not valid_aspects:
        st.warning("⚠️ No valid aspects with at least three sentiment scores to plot.")
        return

    values = [aspect_values[aspect] for aspect in valid_aspects]
    values += values[:1]
    num_vars = len(valid_aspects)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i in range(len(values) - 1):
        ax.plot(angles[i:i+2], values[i:i+2], color=get_colour(values[i]), linewidth=2)

    ax.fill(angles, values, color='lightblue', alpha=0.2)

    for angle, value in zip(angles, values):
        ax.plot(angle, value, 'o', markersize=8, color=get_colour(value))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(valid_aspects, fontsize=12, fontweight='bold')

    label_padding = 0.15
    for angle, aspect, value in zip(angles[:-1], valid_aspects, values[:-1]):
        label_angle = np.degrees(angle)
        ha, va = 'center', 'center'
        x, y = angle, value
        if 0 <= label_angle < 90 or 270 < label_angle <= 360:
            ha = 'left'
            va = 'bottom'
            x += label_padding
            y += label_padding * 0.2
        elif 90 <= label_angle < 180:
            ha = 'left'
            va = 'top'
            x += label_padding
            y -= label_padding * 0.2
        elif 180 <= label_angle < 270:
            ha = 'right'
            va = 'top'
            x -= label_padding
            y -= label_padding * 0.2
        else:
            va = 'bottom'
            y += label_padding

        ax.text(x, y, f"{value:.2f}", ha=ha, va=va, fontsize=10, fontweight="bold", color="black")

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0, 1.01, 0.2)], fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    title = f"Aspect Sentiment Distribution for '{keyword}' in subreddit '{subreddit or 'all'}'"
    plt.title(title, fontsize=14, fontweight='bold', y=1.1)

    # Create custom legend handles and labels in descending order of positivity
    sentiment_levels = [
        (0.8, "Very Positive"),
        (0.6, "Positive"),
        (0.4, "Neutral"),
        (0.2, "Negative"),
        (0.0, "Very Negative")
    ]
    handles = [mpatches.Patch(facecolor=get_colour(value), edgecolor='black', linewidth=0.5, label=label)
               for value, label in sentiment_levels]


    ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.2, 0),
              ncol=1, fontsize='small', frameon=True, title="Sentiment Scale")

    st.pyplot(fig)

def display_aspect_contribution_to_sentiment(aspects):
    if st.session_state['df_comments'].empty:
        st.info("Run analysis first to see aspect contributions.")
        return

    df = st.session_state['df_comments'].copy()
    df['Overall Sentiment Label'] = df['Sentiment Score'].apply(map_sentiment_to_label)

    sentiment_aspect_counts = df.melt(
        id_vars='Overall Sentiment Label',
        value_vars=aspects,
        var_name='Aspect',
        value_name='Aspect Score'
    ).dropna(subset=['Aspect Score'])
    sentiment_aspect_counts['Aspect Sentiment Label'] = sentiment_aspect_counts['Aspect Score'].apply(map_sentiment_to_label)

    grouped_counts = sentiment_aspect_counts.groupby(['Overall Sentiment Label', 'Aspect']).size().unstack(fill_value=0)

    sentiment_order_dropdown = ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]
    available_sentiments = [s for s in sentiment_order_dropdown if s in grouped_counts.index]
    selected_sentiment = st.selectbox("Select a Sentiment Category:", available_sentiments)

    if selected_sentiment:
        sentiment_data = grouped_counts.loc[selected_sentiment]
        valid_aspects = sentiment_data[sentiment_data >= 3]
        if valid_aspects.empty:
            st.warning(f"Not enough data for the '{selected_sentiment}' sentiment category with the current filtering. Each aspect needs at least three mentions.")
        else:
            # Define a color mapping for sentiment labels
            sentiment_color_map = {
                "Very Positive": "forestgreen",
                "Positive": "limegreen",
                "Neutral": "gold",
                "Negative": "salmon",
                "Very Negative": "firebrick",
            }

            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot each aspect with the color corresponding to the selected sentiment
            for aspect, count in valid_aspects.items():
                ax.bar(aspect, count, color=sentiment_color_map.get(selected_sentiment, 'grey')) # Default to grey if sentiment not found

            ax.set_xlabel("Aspect")
            ax.set_ylabel("Number of Comments")
            ax.set_title(f"Aspect Contribution to '{selected_sentiment}' Sentiment (>= 3 mentions)")
            ax.yaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)