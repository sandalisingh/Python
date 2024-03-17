import pandas as pd
# from EmotionClassifier import EmotionClassifier

# Load the dataset
dataset_path = "Conversation_org.csv"
df = pd.read_csv(dataset_path)

# # Keep only 'chat_text' and 'text_response' columns
# df = df[['chat_text', 'text_response']]

# # Instantiate the EmotionClassifier
# emotion_classifier = EmotionClassifier()

# # Predict emotions for each row
# def predict_emotion(text):
#     return emotion_classifier.predict(text)

# df['emotion'] = df['chat_text'].apply(predict_emotion)

# # Reorder columns
# df = df[['chat_text', 'emotion', 'text_response']]

# # Save the modified dataset
# df.to_csv(dataset_path, index=False)

# print("Dataset saved successfully.")
print("\nCounts for each emotion:")
print(df['emotion'].value_counts())
