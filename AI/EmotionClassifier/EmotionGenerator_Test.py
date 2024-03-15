from EmotionGenerator import EmotionGenerator  
from States import logging, EmotionStates, Range  

personality_vector = [9, 0, 5, 1, 10]

environment = input("Environment = ")

# Create an EmotionManager instance
emotion_generator = EmotionGenerator(personality_vector, environment)

# Print the initial emotion
print("-> Initial emotion:", emotion_generator.get_current_emotion_as_string())
print("-> Initial emoji:", emotion_generator.get_current_emotion_emoji())


while(True):
    # Update emotion based on chat text
    emotion_generator.update_emotion(input("Chat = "))

    # Print the updated emotion
    print("-> Updated emotion:", emotion_generator.get_current_emotion_as_string())
    print("-> Updated emoji:", emotion_generator.get_current_emotion_emoji())
