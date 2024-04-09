import pandas as pd
from EmotionClassifier import EmotionClassifier

# Load the dataset
dataset_path = "Dataset/new_2.csv"
data = pd.read_csv(dataset_path)

# Keep only the necessary column and rename it
# data = data[['Best Generated Conversation']]
# data.columns = ['chat_text']

# # Create a new column text_response which contains the chat_text of the next row
# data['text_response'] = data['chat_text'].shift(-1)

# # Remove the last row
# data = data.dropna()

# df = pd.DataFrame(columns=['chat_text', 'text_response'])

# Iterate through each conversation in the dataset
# for conversation in data:
#     print(conversation)
#     if(conversation=='Best Generated Conversation'):
#         continue
#     # Split the conversation into individual lines
#     lines = conversation.strip().split('\n')
#     print(lines)
    
#     # Extract User 1 and User 2 turns alternatively
#     for i in range(0, len(lines), 2):
#         print(i)
#         print(lines[i].split('User 1: '))
#         print(lines[i+1].split('User 2: '))
#         user1 = lines[i].split('User 1: ')[1]  # Extract User 1's message
#         user2 = lines[i+1].split('User 2: ')[1]  # Extract User 2's response
        
#         # Append the extracted conversation to the DataFrame
#         df = df.append({'chat_text': user1, 'text_response': user2}, ignore_index=True)

# Instantiate the EmotionClassifier
emotion_classifier = EmotionClassifier()

# Predict emotions for each row
def predict_emotion(text):
    return emotion_classifier.predict(text)

data['emotion'] = data['text_response'].apply(predict_emotion)

# Reorder columns
data = data[['chat_text', 'emotion', 'text_response']]

# Save the modified dataset
data.to_csv('Dataset/persona_dataset_training_processed.csv', index=False)

print("Dataset saved successfully.")
print("\nCounts for each emotion:")
print(data['emotion'].value_counts())
