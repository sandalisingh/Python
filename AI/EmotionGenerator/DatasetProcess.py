import pandas as pd
from EmotionClassifier import EmotionClassifier

# Load the dataset
dataset_path = "Dataset/Topical_Chat.csv"
data = pd.read_csv(dataset_path)

# Keep only the necessary column and rename it
data = data[['message']]
data.columns = ['chat_text']

# Create a new column text_response which contains the chat_text of the next row
data['text_response'] = data['chat_text'].shift(-1)

# Remove the last row
data = data.dropna()

# Instantiate the EmotionClassifier
emotion_classifier = EmotionClassifier()

# Predict emotions for each row
def predict_emotion(text):
    return emotion_classifier.predict(text)

data['emotion'] = data['text_response'].apply(predict_emotion)

# Reorder columns
data = data[['chat_text', 'emotion', 'text_response']]

# # Save the modified dataset
data.to_csv(dataset_path, index=False)

# print("Dataset saved successfully.")
print("\nCounts for each emotion:")
print(data['emotion'].value_counts())
