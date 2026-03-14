import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

SEED = 42 #aa seed value set karva thi reproducibility aave che je random nakhyu chhe
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# IMDB data loading 

max_features = 10000
maxlen = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


#LSTM model (anathi aagad vadhavanu chhe)

model = Sequential([
    Embedding(max_features, 32),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)


model.save("lstm_model.keras")
print("✅ Model saved as lstm_model.keras")
