from States import logging, EmotionStates
from prettytable import PrettyTable
from Tokenizer import Tokenizer
from DataVisualizer import DataVisualizer
from sklearn.utils import resample
import tensorflow as tf
import numpy as np
import pandas as pd
import string
import re

class DataManager:
    
    @staticmethod
    def load_dataframe(dataset_mame):
        df = pd.read_csv(dataset_mame)
        # DataVisualizer.display_top_rows(df)
        logging("info","Dataframe loaded.")
        return df
    
    @staticmethod
    def extract_data_from_dataframe(df):
        chat_text = df['chat_text'].tolist()
        text_response = df['text_response'].tolist()
        emotion = df['emotion'].tolist()
        logging("info","Data extracted from Dataframe.")
        DataManager.count_emotions(emotion)
        return chat_text, text_response, emotion

    @staticmethod
    def count_emotions(emotion):
        unique_emotions = set(emotion)
        total_count = len(emotion)

        print("Emotion labels count:")
        for emo in unique_emotions:
            count = emotion.count(emo)
            print(f"{emo}: {count}")
            
        print(f"Total count: {total_count}")
    
    @staticmethod
    def preprocess_data(chat_text, text_response, emotion, VOCAB_SIZE, MAX_SEQ_LENGTH):
        # print("\n\n-> PREPROCESS DATA")

        # DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : chat_text, 'emotion' : emotion, 'text_response' : text_response}), 1, "Input")

        # Padding punctuation marks
        chat_text = [DataManager.pad_punctuation(text) for text in chat_text]
        text_response = [DataManager.pad_punctuation(text) for text in text_response]

        # DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : chat_text, 'emotion' : emotion, 'text_response' : text_response}), 1, "Padding punctuation marks")

        tokenizer = Tokenizer(VOCAB_SIZE)
        tokenizer.fit_tokenizer(chat_text, text_response)

        chat_text = tokenizer.TOKENIZER.texts_to_sequences(chat_text)
        text_response = tokenizer.TOKENIZER.texts_to_sequences(text_response)
        # DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : chat_text, 'emotion' : emotion, 'text_response' : text_response}), 1, "Token to index")

        # Add <end> token to each chat text sequence
        for seq in chat_text:
            seq.append(tokenizer.END_TOKEN)
        
        # Add <start> token to each text response sequence
        for seq in text_response:
            seq.insert(0, tokenizer.START_TOKEN)
        
        # Add <end> token to each text response sequence
        for seq in text_response:
            seq.append(tokenizer.END_TOKEN)

        # DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : chat_text, 'emotion' : emotion, 'text_response' : text_response}), 1, "Marking start and end of sequences")

        # Pad sequences
        chat_text = tf.keras.preprocessing.sequence.pad_sequences(chat_text, maxlen=MAX_SEQ_LENGTH, padding='post')
        text_response = tf.keras.preprocessing.sequence.pad_sequences(text_response, maxlen=MAX_SEQ_LENGTH, padding='post')
        # DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : chat_text.tolist(), 'emotion' : emotion, 'text_response' : text_response.tolist()}), 1, "Padding sequences for constant frame length")

        # Shift targets for teacher forcing
        output_seq = np.zeros_like(text_response)
        output_seq[:, :-1] = text_response[:, 1:]                                    
        output_seq[:, -1] = 0

        # Map emotion strings to their corresponding enum values
        emotion = [EmotionStates.string_to_index(emo) for emo in emotion] 
        # DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : chat_text.tolist(), 'emotion' : emotion, 'text_response' : text_response.tolist()}), 1, "Emotion to index")

        # convert list to vector
        emotion = np.array(emotion)
        emotion = np.expand_dims(emotion, axis=1)
        
        logging("info","Data preprocessed.")

        chat_text = tf.convert_to_tensor(chat_text, dtype=tf.float32)
        emotion = tf.convert_to_tensor(emotion, dtype=tf.float32)
        text_response = tf.convert_to_tensor(text_response, dtype=tf.float32)
        output_seq = tf.convert_to_tensor(output_seq, dtype=tf.float32)

        # DataVisualizer.print_tensor_dict("Model definition: Inputs and Outputs", {'chat_text_input' : chat_text, 'emotion_input' : emotion, 'prev_seq_input' : text_response, 'output_seq' : output_seq})
        
        return chat_text, emotion, text_response, output_seq

    @staticmethod
    def pad_punctuation(text):
        punctuation = string.punctuation.replace("'", "")
        punctuation = punctuation.replace("-", "")
        return re.sub(f"([{punctuation}])", r" \1 ", text)

    @staticmethod
    def prepare_data(dataset_mame):
        df = DataManager.load_dataframe(dataset_mame)
        
        # Extract data from the DataFrame
        chat_text, text_response, emotion = DataManager.extract_data_from_dataframe(df)
        
        return chat_text, text_response, emotion
        