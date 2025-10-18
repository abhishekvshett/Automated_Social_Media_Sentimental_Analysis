# 🎯 Automated Social Media Sentiment Analysis (YouTube Comments)

## 🧠 Overview
This project performs **sentiment analysis on YouTube comments** using a **BERT-based Deep Learning model**.  
It fetches comments from a video via the **YouTube Data API**, analyzes each comment’s sentiment, and visualizes the overall sentiment distribution through an **interactive Streamlit dashboard**.

The system helps creators, brands, and analysts monitor **viewer emotions** and **brand perception** efficiently.

---

## 🚀 Features
✅ Fetches YouTube comments in real time using the **YouTube Data API**  
✅ Converts **UTC timestamps to IST** for Indian time-based trend visualization  
✅ Uses **BERT (nlptown/bert-base-multilingual-uncased-sentiment)** for multilingual sentiment detection  
✅ Generates:
- 📊 **Bar Chart** → Frequency of sentiment types  
- 📈 **Line Chart** → Sentiment change over time  
- 🕐 **Hourly Sentiment Trend**  
- 🥧 **Pie Chart** → Sentiment proportion  
✅ Builds a **Word Cloud** of most common words  
✅ Fully interactive dashboard using **Streamlit** and **Plotly**

---

## 🧰 Tech Stack
| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Machine Learning Model** | BERT (Hugging Face Transformers) |
| **Visualization** | Plotly, Matplotlib, WordCloud |
| **Data Source** | YouTube Data API |
| **Libraries** | `transformers`, `torch`, `googleapiclient`, `pandas`, `plotly`, `streamlit`, `matplotlib`, `wordcloud` |

---

## ⚙️ How It Works

### 1. Fetch YouTube Comments  
- The app uses `googleapiclient.discovery` to fetch top-level comments using a **YouTube video ID** and **API key**.  
- Comments, author names, timestamps, and like counts are stored in a DataFrame.

### 2. Preprocess & Convert Time  
- Converts timestamps from **UTC → Indian Standard Time (IST)** using `pytz`.

### 3. Sentiment Classification  
- Uses the pre-trained **BERT model** (`nlptown/bert-base-multilingual-uncased-sentiment`) to classify each comment into one of five categories:  
  | Score | Sentiment |
  |--------|------------|
  | 1 | Awful |
  | 2 | Bad |
  | 3 | Neutral |
  | 4 | Good |
  | 5 | Excellent |

### 4. Visualization Dashboard  
- Interactive Streamlit dashboard displaying:
  - Sentiment-wise comment distribution  
  - Word cloud of most used words  
  - Sentiment trends over **time** and **hour of day**  
  - Proportion of positive, neutral, and negative comments  

---

## 🧪 How to Run the Project

### 🔧 Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/social-media-sentiment-analysis.git
cd social-media-sentiment-analysis  
```
###🔧 Step 2: Install Dependencies

    pip install -r requirements.txt

### 🗝️ Step 3: Get YouTube API Key

 -Visit Google Cloud Console

 -Enable YouTube Data API v3

 -Generate an API key

 -Paste it into the Streamlit input field when prompted

### ▶️ Step 4: Run the Streamlit App

    streamlit run app.py
    
### 🧭 Step 5: Enter Details in the Web App

 -Input your YouTube video ID (from the video URL)

-Example:
 -For https://www.youtube.com/watch?v=abcd1234, the ID is abcd1234

 -Enter your YouTube API Key

 -Click "Analyse Comments"

 ---

 ## 📸 Screenshots

 <img width="1900" height="1008" alt="Screenshot 2025-10-17 141158" src="https://github.com/user-attachments/assets/d3833bfe-e6fe-4782-9407-884a5c803368" />
 
 ---
 
 <img width="1906" height="970" alt="Screenshot 2025-10-18 101433" src="https://github.com/user-attachments/assets/21d1bd19-1a74-4760-bc3c-ed6ac6249f3a" />
 
 ---
 
 <img width="1916" height="776" alt="Screenshot 2025-10-18 101453" src="https://github.com/user-attachments/assets/4d97c9e5-047f-448a-ae7e-111fbb06c323" />

 ---
 <img width="1162" height="770" alt="Screenshot 2025-10-18 101520" src="https://github.com/user-attachments/assets/5fe896ad-8018-4448-a70e-5fff2e6807f3" /> 

 ---
 
 <img width="1910" height="856" alt="Screenshot 2025-10-18 101549" src="https://github.com/user-attachments/assets/fa2878f3-54fe-4093-bae0-e9421bd5f205" />

 ---
 
<img width="1169" height="761" alt="Screenshot 2025-10-18 101746" src="https://github.com/user-attachments/assets/6570b2a7-c6f2-4772-a0e2-0f6a0c7a8093" /> 

---

<img width="1902" height="957" alt="Screenshot 2025-10-18 101801" src="https://github.com/user-attachments/assets/60c1be89-a20b-44e5-9b42-79a36f2191bc" />

---










