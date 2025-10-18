import googleapiclient.discovery
import pandas as pd
import pytz
from datetime import datetime
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Prevent PyTorch-Streamlit compatibility issue
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Import PyTorch and transformers after setting environment variable
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Function to convert UTC timestamp to IST and format
def convert_to_indian_time(timestamp):
    utc_time = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
    utc_time = pytz.utc.localize(utc_time)
    ist_time = utc_time.astimezone(pytz.timezone('Asia/Kolkata'))
    return ist_time.strftime('%d/%m/%Y %H:%M:%S')

# Function to fetch comments from YouTube with pagination
def fetch_youtube_comments(video_id, api_key, max_results=300):
    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=api_key)

    comments = []
    next_page_token = None
    while len(comments) < max_results:
        # Request to fetch comment threads
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results - len(comments), 100),  # Limit to remaining needed comments or 100, whichever is smaller
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['textDisplay'],
                comment['authorDisplayName'],
                convert_to_indian_time(comment['updatedAt']),
                comment['likeCount']
            ])

        if 'nextPageToken' in response:
            next_page_token = response['nextPageToken']
        else:
            break

    df = pd.DataFrame(comments[:max_results], columns=['Comment', 'Author', 'Last updated at', 'Like count'])
    return df

# Function for sentiment analysis
def sentiment_analysis(text):
    try:
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512)
        result = model(tokens)
        sentiment_score = int(torch.argmax(result.logits)) + 1
        likert_scale = {1: "Awful", 2: "Bad", 3: "Neutral", 4: "Good", 5: "Excellent"}
        return likert_scale[sentiment_score]
    except Exception as e:
        return "Unknown"

# Function to generate Word Cloud
def generate_wordcloud(df):
    text = ' '.join(df['Comment'].values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Create a figure and display it properly with Streamlit
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Use st.pyplot with the figure object
    st.pyplot(fig)

st.title("Social Media Sentiment Analysis")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["YouTube Comments", "Playstore Review", "X Threads"])

if page == "YouTube Comments":
    st.write("## YouTube Comments Analysis")
    st.write("#### Enter a YouTube video ID and API key to fetch comments and perform sentiment analysis.")

    video_id = st.text_input("Enter YouTube Video ID:")
    api_key = st.text_input("Enter YouTube API Key:", type="password")

    if st.button("Analyse Comments"):
        if video_id and api_key:
            placeholder_fetching = st.empty()
            try:
                # Fetch comments using the function
                placeholder_fetching.text("Fetching comments...")
                df = fetch_youtube_comments(video_id, api_key, max_results=300)

                # Check if any comments were fetched
                if df.empty:
                    st.warning("No comments fetched. Please check the video ID or try again later.")

                # Perform sentiment analysis
                placeholder_fetching.text("Performing sentiment analysis...")
                df['Sentiment'] = df['Comment'].apply(sentiment_analysis)

                # Generating visualizations
                placeholder_fetching.text("Generating visualizations...")
                st.write(f"Number of Comments Fetched: {len(df)} most recent comments")

                # Display DataFrame with Sentiment column
                st.write("### Comments DataFrame:")
                st.dataframe(df[['Comment', 'Sentiment', 'Author', 'Last updated at', 'Like count']])

                # Word Cloud and Explanation
                st.write("### Sentiments Word Cloud:")
                st.write("A Word Cloud representation of most used words in the comments: ")
                generate_wordcloud(df)

                # Plot frequency of sentiment comments
                st.write("### Sentiment Frequency:")
                sentiment_order = ["Awful", "Bad", "Neutral", "Good", "Excellent"]
                sentiment_counts = df['Sentiment'].value_counts().reindex(sentiment_order, fill_value=0)
                sentiment_df = pd.DataFrame({
                    'Sentiment': sentiment_order,
                    'Count': sentiment_counts
                })
                fig = px.bar(sentiment_df, x='Sentiment', y='Count', color='Sentiment',
                             color_discrete_sequence=px.colors.qualitative.Plotly,
                             category_orders={'Sentiment': sentiment_order},
                             title='Frequency of Sentiment Comments',
                             labels={'Count': 'Number of Comments', 'Sentiment': 'Sentiment'},
                             template='plotly')
                st.plotly_chart(fig)
                
                # Inference
                st.write("**Inference:** The bar chart shows the distribution of sentiments among the comments. ")

                

                # Plot line graph of how total number of comments related to a sentiment is changing over time
                st.write("### Sentiment Distribution Over Time:")
                df['Date'] = pd.to_datetime(df['Last updated at'], format='%d/%m/%Y %H:%M:%S').dt.date
                sentiments_by_date = df.groupby(['Date', 'Sentiment']).size().unstack().fillna(0)
                sentiments_by_date = sentiments_by_date.reindex(columns=sentiment_order, fill_value=0)
                fig_sentiment_over_time = px.line(sentiments_by_date, x=sentiments_by_date.index, y=sentiments_by_date.columns,
                                                  title='Sentiment Distribution Over Time',
                                                  labels={'value': 'Number of Comments', 'Date': 'Date'},
                                                  template='plotly')
                st.plotly_chart(fig_sentiment_over_time)

                # Inference
                st.write("**Inference:** This line graph illustrates how the distribution of sentiments evolves over time. ")

                # Plot a pie chart representing proportion of sentiments
                st.write("### Proportion of Sentiments:")
                sentiment_proportions = df['Sentiment'].value_counts(normalize=True)
                fig_pie_sentiments = px.pie(values=sentiment_proportions, names=sentiment_proportions.index,
                                           title='Proportion of Sentiments',
                                           labels={'Sentiment': 'Sentiment', 'value': 'Proportion'},
                                           template='plotly')
                st.plotly_chart(fig_pie_sentiments)
                # Inference
                st.write("**Inference:** The pie chart shows the proportion of different sentiments among the comments. ")

                # Plot line graph of how total number of comments related to a sentiment is changing over time
                st.write("### Sentiment Distribution by Hour of the Day:")
                try:
                    df['Hour'] = pd.to_datetime(df['Last updated at'], format='%d/%m/%Y %H:%M:%S').dt.hour
                except KeyError:
                    st.error("The 'Last updated at' column does not exist. Please ensure that comments have been fetched correctly.")
                    st.stop()
                except Exception as e:
                    st.error(f"An error occurred while processing the 'Last updated at' column: {e}")
                    st.stop()
                
                sentiments_by_hour = df.groupby(['Hour', 'Sentiment']).size().unstack().fillna(0)
                sentiments_by_hour = sentiments_by_hour.reindex(columns=sentiment_order, fill_value=0)
                fig_hourly_sentiment = px.line(sentiments_by_hour, x=sentiments_by_hour.index, y=sentiments_by_hour.columns,
                                               title='Sentiment Distribution by Hour of the Day',
                                               labels={'value': 'Number of Comments', 'Hour': 'Hour of the Day'},
                                               template='plotly')
                st.plotly_chart(fig_hourly_sentiment)

                # Inference
                st.write("**Inference:** The line graph shows how the distribution of sentiments changes throughout the day. For example, if any sentiment peaks at the specific time, it might suggest that viewers are more prone to that sentiment during that time of the day. ")

                placeholder_fetching.text("Analysis completed!")

            except Exception as e:
                st.error(f"Error: {e}")

        else:
            st.warning("Please enter both YouTube Video ID and API Key.")
