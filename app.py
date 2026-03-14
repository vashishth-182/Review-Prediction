import re
import json
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

@st.cache_data
def load_word_index():
    with open("imdb_word_index.json", "r") as f:
        word_index = json.load(f)
    word_to_id = {k: (v + 3) for k, v in word_index.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_id["<UNUSED>"] = 3
    return word_to_id

word_to_id = load_word_index()

def preprocess_text(text, maxlen=500):
    # Remove punctuation so 'movie!' doesn't become 'movie!' (which isn't in vocab)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.lower().split()
    seq = []

    for word in words:
        idx = word_to_id.get(word, 2)  # 2 vadu yad hatu hal bhulo gyo
        if idx >= 10000:
            idx = 2
        seq.append(idx)

    return sequence.pad_sequences([seq], maxlen=maxlen)


@st.cache_resource
def get_model():
    # Cache completely cleared to load the official Keras 3 model.
    return load_model("lstm_model.keras")

try:
    model = get_model() 
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize session state for review history
if 'history' not in st.session_state:
    st.session_state['history'] = []

st.title("🧠 LSTM Sentiment Analyzer")
st.write("Enter a MOVIE REVIEW to predict if it's Positive or Negative.")

# Sidebar for History
with st.sidebar:
    st.header("🕒 Recent History")
    if not st.session_state['history']:
        st.write("No predictions yet.")
    else:
        for item in reversed(st.session_state['history']):
            st.markdown(f"**{item['sentiment']}** (`{item['score']:.2f}`)")
            st.caption(f'"{item["text"][:60]}..."')
            st.divider()

        if st.button("Clear History"):
            st.session_state['history'] = []
            st.rerun()

tab1, tab2 = st.tabs(["✍️ Single Review", "📁 Batch Processing (CSV / TXT)"])

with tab1:
    user_input = st.text_area("Write your review here:")

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review text.")
        else:
            try:
                data = preprocess_text(user_input)
                prediction = model.predict(data, verbose=0)[0][0]
                
                is_positive = prediction > 0.5
                sentiment = "🙂 Positive" if is_positive else "☹️ Negative"
                
                st.markdown(f"### Prediction: **{sentiment}**")
                
                # Visual Meter
                st.progress(float(prediction), text=f"Confidence Score: {prediction:.4f}")
                
                # Save to history
                st.session_state['history'].append({
                    "text": user_input,
                    "sentiment": sentiment,
                    "score": float(prediction)
                })
            except Exception as e:
                st.error(f"Error making prediction: {e}")

with tab2:
    st.markdown("Upload a CSV or TXT file containing reviews to analyze them all at once.")
    uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                # Process TXT file (each non-empty line is treated as a review)
                content = uploaded_file.getvalue().decode("utf-8")
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                df = pd.DataFrame(lines, columns=["Review Text"])

            st.write("Preview of uploaded data:")
            st.dataframe(df.head(3))
            
            column_to_analyze = st.selectbox("Select the column containing the reviews:", df.columns)
            
            if st.button("Analyze Batch"):
                with st.spinner('Analyzing reviews...'):
                    predictions = []
                    sentiments = []
                    for text in df[column_to_analyze].astype(str):
                        data = preprocess_text(text)
                        pred = model.predict(data, verbose=0)[0][0]
                        predictions.append(pred)
                        sentiments.append("Positive" if pred > 0.5 else "Negative")
                        
                    df['Sentiment Score'] = predictions
                    df['Predicted Sentiment'] = sentiments
                    
                    st.success("✅ Batch Analysis Complete!")
                    st.dataframe(df)
                    
                    st.subheader("Results Summary")
                    sentiment_counts = df['Predicted Sentiment'].value_counts()
                    st.bar_chart(sentiment_counts)
                    
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
