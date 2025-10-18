🎯 Automated Social Media Sentiment Analysis 

<br>
🧠 Overview

This project performs sentiment analysis on YouTube/Twitter comments using a BERT-based Deep Learning model.
It fetches comments from a video via the YouTube Data API, analyzes each comment’s sentiment, and visualizes the overall sentiment distribution through an interactive Streamlit dashboard.

The system helps creators, brands, and analysts monitor viewer emotions and brand perception efficiently.
<br>
<hr>
🚀 Features <br>

✅ Fetches YouTube comments in real time using the YouTube Data API <br>
✅ Converts UTC timestamps to IST for Indian time-based trend visualization <br>
✅ Uses BERT (nlptown/bert-base-multilingual-uncased-sentiment) for multilingual sentiment detection <br>
✅ Generates:

   📊 Bar Chart → Frequency of sentiment types

   📈 Line Chart → Sentiment change over time

   🕐 Hourly Sentiment Trend

   🥧 Pie Chart → Sentiment proportion
   ✅ Builds a Word Cloud of most common words
   ✅ Fully interactive dashboard using Streamlit and Plotly
<br>
<hr>
🧰 Tech Stack <br>
Component--->	Technology <br>
Frontend--->	Streamlit <br>
Backend -->	Python <br>
Machine Learning Model--->	BERT (Hugging Face Transformers) <br>
Visualization--->	Plotly, Matplotlib, WordCloud 
<br>
<hr>
⚙️ How It Works
<br>
1. Fetch YouTube Comments

  The app uses googleapiclient.discovery to fetch top-level comments using a YouTube video ID and API key.

  Comments, author names, timestamps, and like counts are stored in a DataFrame.

2. Preprocess & Convert Time

  Converts timestamps from UTC → Indian Standard Time (IST) using pytz.

3. Sentiment Classification

  Uses the pre-trained BERT model (nlptown/bert-base-multilingual-uncased-sentiment) to classify each comment into one of five categories:

Score	Sentiment
    1	Awful
    2	Bad
    3	Neutral
    4	Good
    5	Excellent
    
4. Visualization Dashboard

  Interactive Streamlit dashboard displaying:

  Sentiment-wise comment distribution

  Word cloud of most used words

  Sentiment trends over time and hour of day

  Proportion of positive, neutral, and negative comments

  <br>


