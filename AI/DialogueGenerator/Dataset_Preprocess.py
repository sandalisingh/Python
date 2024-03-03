import pandas as pd
from joblib import load

# Load the dataset
dataset = pd.read_csv("Conversation2.csv")

# Rename columns
# dataset.rename(columns={"question": "chat_text", "answer": "text_response"}, inplace=True)

# Load the emotion classifier model
emotion_classifier_model = load("emotion_classifier_model.joblib")

# # Emotion mapping dictionary
# emotion_mapping = {
#     0: 'admiration',
#     1: 'amusement',
#     2: 'anger',
#     3: 'annoyance',
#     4: 'approval',
#     5: 'caring',
#     6: 'confusion',
#     7: 'curiosity',
#     8: 'desire',
#     9: 'disappointment',
#     10: 'disapproval',
#     11: 'disgust',
#     12: 'embarrassment',
#     13: 'excitement',
#     14: 'fear',
#     15: 'gratitude',
#     16: 'grief',
#     17: 'joy',
#     18: 'love',
#     19: 'nervousness',
#     20: 'optimism',
#     21: 'pride',
#     22: 'realization',
#     23: 'relief',
#     24: 'remorse',
#     25: 'sadness',
#     26: 'surprise',
#     27: 'neutral'
# }

# Function to predict emotion index
def predict_emotion(text):
    # Assuming you have a function to preprocess text if needed
    # preprocessed_text = preprocess(text)
    # Assuming your model takes preprocessed text as input
    # emotion_index = emotion_classifier_model.predict(preprocessed_text)
    emotion = emotion_classifier_model.predict([text])[0]
    return emotion

# # Function to map emotion index to emotion
# def map_emotion(emotion_index):
#     return emotion_mapping.get(emotion_index, 'unknown')

# Apply the model to the "text_response" column
dataset['emotion'] = dataset['text_response'].apply(predict_emotion)

# # Map emotion index to emotion
# dataset['emotion'] = dataset['emotion_index'].apply(map_emotion)

# Reorder the columns
dataset = dataset[['chat_text', 'emotion', 'text_response']]

# Save the updated dataset
dataset.to_csv("Conversation3.csv", index=False)

# # Load the dataset
# dataset = pd.read_csv("updated_Conversation_dataset.csv")

# # Rename the column "Unnamed: 0" to "S.No."
# dataset.rename(columns={"Unnamed: 0": "S.No."}, inplace=True)

# # Drop the "emotion_index" column
# dataset.drop(columns=["emotion_index"], inplace=True)

# # Save the updated dataset
# dataset.to_csv("Conversation2.csv", index=False)

# Assuming your dataset is in a CSV file named 'dataset.csv'
# # Load the dataset
# df = pd.read_csv('Conversation3.csv')

# # Reorder the columns
# df = df[['S.No.', 'chat_text', 'emotion', 'text_response']]

# # Save the updated dataset to a new CSV file or overwrite the existing one
# df.to_csv('Conversation2.csv', index=False)
