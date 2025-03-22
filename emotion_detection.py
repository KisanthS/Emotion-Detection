# Step 1: Import required libraries
import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 2: Load train, validation, and test datasets
train = pd.read_table('train.txt', delimiter=';', header=None)
val = pd.read_table('val.txt', delimiter=';', header=None)
test = pd.read_table('test.txt', delimiter=';', header=None)

# Combine all datasets
data = pd.concat([train, val, test])
data.columns = ["text", "label"]

# Step 2.1: Text Preprocessing Function
ps = PorterStemmer()
def preprocess(line):
    review = re.sub('[^a-zA-Z]', ' ', line).lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    return " ".join(review)

# Apply preprocessing
data['text'] = data['text'].apply(preprocess)

# Step 3: Convert text into numerical data using improved TfidfVectorizer
tfidf = TfidfVectorizer(max_features=7000, ngram_range=(1,3), stop_words="english")
data_tfidf = tfidf.fit_transform(data['text']).toarray()

# Step 4: Encode Labels
label_encoder = preprocessing.LabelEncoder()
data['N_label'] = label_encoder.fit_transform(data['label'])

# Step 5: Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(data_tfidf, data['N_label'], test_size=0.25, random_state=42)

# Step 6: Build the Neural Network Model
model = Sequential([
    Dense(16, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(6, activation='softmax')  # 6 emotions in dataset
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model (Increased epochs)
model.fit(X_train, y_train, epochs=20, batch_size=8)

# Step 7: Evaluate & Save Model
_, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save model, vectorizer, and encoder
model.save('my_model.h5')
pickle.dump(label_encoder, open('encoder.pkl', 'wb'))
pickle.dump(tfidf, open('TfidfVectorizer.pkl', 'wb'))
