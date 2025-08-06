import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

word_index = imdb.get_word_index()
word_to_id = {k: (v + 3) for k, v in word_index.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id["<UNUSED>"] = 3

def preprocess_text(text, maxlen=500):
    words = text.lower().split()
    seq = []

    for word in words:
        idx = word_to_id.get(word, 2)  # 2 vadu yad hatu hal bhulo gyo
        if idx >= 10000:
            idx = 2
        seq.append(idx)

    return sequence.pad_sequences([seq], maxlen=maxlen)


try:
    model = load_model("lstm_model.h5") 
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# apapdu streamlit 

st.title("🧠 LSTM Sentiment Analyzer")
st.write("Enter a MOVIE REVIEW to predict if it's Positive or Negative.")

user_input = st.text_area("✍️ Write your review here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review text.")
    else:
        try:
            data = preprocess_text(user_input)
            prediction = model.predict(data)[0][0]
            sentiment = "🙂 Positive" if prediction > 0.5 else "☹️ Negative"
            st.markdown(f"### Prediction: **{sentiment}**")
            st.markdown(f"**Confidence Score:** `{prediction:.4f}`")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
