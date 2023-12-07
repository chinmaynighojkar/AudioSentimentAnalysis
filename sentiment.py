#pip install streamlit
#pip install git+https://github.com/openai/whisper.git 

import streamlit as st 
import whisper
import pandas as pd
import seaborn as sns
import matplotlib as pyplot
from matplotlib import pyplot as plt
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')  # Download the VADER lexicon if you haven't already
analyzer = SentimentIntensityAnalyzer()

st.title("Whisper App")
st.text("Whisper Base Model Loaded")

# upload audio file with streamlit
model= whisper.load_model("base")
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

st.header("Play Original Audio File")
st.audio (audio_file)


if st.button("Transcribe Audio"):
    if audio_file is not None:
        st.success("Transcribing Audio")
        result= model.transcribe (audio_file.name)

        st.success("Transcription Complete")
        st.markdown (result["text"])

    else:
        st.error("Please upload an audio file")
        
#result['segments']
speech = pd.DataFrame.from_dict(result['segments'])
for count_line, seg in enumerate(result['segments']):
  print(count_line, "- ", seg['text'])

st.title("Getting text into a Dataframe:")
speech = pd.DataFrame.from_dict(result['segments'])
st.write(speech)


df = speech
#st.title("Input your CSV File")
#st.write("The CSV File should have columns id and text")
#st.write("Upload a CSV file and view its contents")

#uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
#if uploaded_file is not None:
#  df = pd.read_csv(uploaded_file)
#  st.write("File Contents:")
#  df.dropna(inplace=True)
#  st.write(df)

#st.title("Reviewing columns id and text with start time and end time:")
#st.write(df)

#df.info()
#df.isnull().sum()
#df.dropna(inplace = True)
#df.isnull().sum()

df['Sentiment_Score'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
def categorize_sentiment(score):
    if score > 0.75:
        return 'Joy'  # Replacing Positive with Joy
    elif score > 0.3:
        return 'Happy'  # Adding a new category for scores between 0 and 0.5
    elif score >= -0.05:
        return 'Neutral'  # Retaining Neutral for scores between -0.05 and 0.05
    elif score > -0.2:
        return 'Sad'  # Adding a new category for scores between -0.05 and -0.1
    else:
        return 'Disgust'  # Replacing Negative with Disgust


df['Predicted_Sentiment'] = df['Sentiment_Score'].apply(categorize_sentiment)

st.title("Analysis with Score and Sentiment:")
st.write(df)



st.title("Analysis with Score and Sentiment:")
st.write(df)



st.title('Sentiment Score Bar Chart')

# Create a bar chart for sentiment categories
# Create a bar chart for sentiment categories with color-coding
sentiment_counts = df['Predicted_Sentiment'].value_counts()

# Define colors for each sentiment category
colors = {
    'Joy': 'green',
    'Happy': 'lightgreen',
    'Neutral': 'grey',
    'Sad': 'lightcoral',
    'Disgust': 'red'
}

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=[colors[sentiment] for sentiment in sentiment_counts.index])
plt.title("Sentiment Category Counts")
plt.xlabel("Sentiment Category")
plt.ylabel("Count")

# Display the bar chart using Streamlit
st.pyplot(plt)


