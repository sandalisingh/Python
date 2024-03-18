import pandas as pd
from EmotionClassifier import EmotionClassifier

# Load the dataset
dataset_path = "Dataset/Ubuntu_Dataset_1_org.csv"
data = pd.read_csv(dataset_path)

# # Keep only 'chat_text' and 'text_response' columns
# df = df[['chat_text', 'text_response']]

# Keep only the 'text' column and rename it to 'chat_text'
data = data[['text']].rename(columns={'text': 'chat_text'})

# Add another column 'text_response' with the value of the next row's 'chat_text'
data['text_response'] = data['chat_text'].shift(-1)

# Remove the last row
data = data[:-1]

# Display the resulting DataFrame
print(data)

# Instantiate the EmotionClassifier
emotion_classifier = EmotionClassifier()

# Predict emotions for each row
def predict_emotion(text):
    return emotion_classifier.predict(text)

df['emotion'] = df['chat_text'].apply(predict_emotion)

# Reorder columns
df = df[['chat_text', 'emotion', 'text_response']]

# # Save the modified dataset
data.to_csv(dataset_path, index=False)

# print("Dataset saved successfully.")
print("\nCounts for each emotion:")
print(df['emotion'].value_counts())
