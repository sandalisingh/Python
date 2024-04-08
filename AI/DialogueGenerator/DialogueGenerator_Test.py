import pickle
from DialogueGenerator import DialogueGenerator
import tensorflow as tf

dialogue_generator = DialogueGenerator()
dialogue_generator.reset_states()

while(True) : 
    dialogue_generator.reset_states()
    text = input("\nUser : ")
    emotion = input("Emotion : ")
    response = dialogue_generator.generate_response_with_greedy_approach(text, emotion)
    print("\n-> Response:", response)
