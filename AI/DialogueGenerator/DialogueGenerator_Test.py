import pickle
from DialogueGenerator import DialogueGenerator
import tensorflow as tf

dialogue_generator = DialogueGenerator()

while(True) :
    text = input("\n$ User : ")
    emotion = input("$ Emotion : ")
    response = dialogue_generator.generate_response_with_beam_search(text, emotion)
    print("\n$ Response:", response)
