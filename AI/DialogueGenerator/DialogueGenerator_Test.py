import pickle
from DialogueGenerator import generate_response, MAX_SEQ_LENGTH, generate_response_beam_search
import tensorflow as tf

# Load the tokenizer
tokenizer_path = "tokenizer.pkl"
with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
print("Tokenizer loaded from ", tokenizer_path)

# Load the trained model
model = tf.keras.models.load_model("conversation_model.keras")
print("Trained model loaded.")

while(True) :
    text = input("\n$ User : ")
    emotion = input("$ Emotion : ")
    response = generate_response(text, emotion, tokenizer, model, MAX_SEQ_LENGTH)
    print("\n$ Response:", response)
