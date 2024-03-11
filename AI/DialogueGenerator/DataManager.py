from States import logging, EmotionStates
from prettytable import PrettyTable
from Tokenizer import Tokenizer
from DataVisualizer import DataVisualizer
import tensorflow as tf
import numpy as np
import pandas as pd
import string
import re

class DataManager:
    
    @staticmethod
    def load_dataframe(dataset_mame):
        df = pd.read_csv(dataset_mame)
        DataVisualizer.display_top_rows(df)
        logging("info","Dataframe loaded.")
        return df
    
    @staticmethod
    def extract_data_from_dataframe(df):
        chat_text = df['chat_text'].tolist()
        text_response = df['text_response'].tolist()
        emotion = df['emotion'].tolist()
        logging("info","Data extracted from Dataframe.")
        return chat_text, text_response, emotion

    @staticmethod
    def preprocess_data(chat_text, text_response, emotion, VOCAB_SIZE, MAX_SEQ_LENGTH):
        print("\n\n-> PREPROCESS DATA")

        DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : chat_text, 'emotion' : emotion, 'text_response' : text_response}), 1, "Input")

        # Padding punctuation marks
        chat_text = [DataManager.pad_punctuation(text) for text in chat_text]
        text_response = [DataManager.pad_punctuation(text) for text in text_response]

        DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : chat_text, 'emotion' : emotion, 'text_response' : text_response}), 1, "Padding punctuation marks")

        tokenizer = Tokenizer(VOCAB_SIZE)
        tokenizer.create_tokenizer(chat_text, text_response)

        chat_text_sequences = tokenizer.TOKENIZER.texts_to_sequences(chat_text)
        text_response_sequences = tokenizer.TOKENIZER.texts_to_sequences(text_response)
        DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : chat_text_sequences, 'emotion' : emotion, 'text_response' : text_response_sequences}), 1, "Token to index")

        # Add <end> token to each chat text sequence
        for seq in chat_text_sequences:
            seq.append(tokenizer.END_TOKEN)
        
        # Add <start> token to each text response sequence
        for seq in text_response_sequences:
            seq.insert(0, tokenizer.START_TOKEN)
        
        # Add <end> token to each text response sequence
        for seq in text_response_sequences:
            seq.append(tokenizer.END_TOKEN)

        DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : chat_text_sequences, 'emotion' : emotion, 'text_response' : text_response_sequences}), 1, "Marking start and end of sequences")

        # Pad sequences
        encoder_inputs_chat_text = tf.keras.preprocessing.sequence.pad_sequences(chat_text_sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
        decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(text_response_sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
        DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : encoder_inputs_chat_text.tolist(), 'emotion' : emotion, 'text_response' : decoder_inputs.tolist()}), 1, "Padding sequences for constant frame length")

        # Shift targets for teacher forcing
        decoder_outputs = np.zeros_like(decoder_inputs)
        decoder_outputs[:, :-1] = decoder_inputs[:, 1:]                                    
        decoder_outputs[:, -1] = 0

        # Map emotion strings to their corresponding enum values
        emotion_inputs = [EmotionStates.get_emotion_index(emo) for emo in emotion] 
        DataVisualizer.display_top_rows(pd.DataFrame({'chat_text' : encoder_inputs_chat_text.tolist(), 'emotion' : emotion_inputs, 'text_response' : decoder_inputs.tolist()}), 1, "Emotion to index")

        # convert list to vector
        emotion_inputs = np.array(emotion_inputs)
        emotion_inputs = np.expand_dims(emotion_inputs, axis=1)
        
        logging("info","Data preprocessed.")

        encoder_inputs_chat_text = tf.convert_to_tensor(encoder_inputs_chat_text, dtype=tf.float32)
        emotion_inputs = tf.convert_to_tensor(emotion_inputs, dtype=tf.float32)
        decoder_inputs = tf.convert_to_tensor(decoder_inputs, dtype=tf.float32)
        decoder_outputs = tf.convert_to_tensor(decoder_outputs, dtype=tf.float32)

        DataVisualizer.print_tensor_dict("Model definition: Inputs and Outputs", {'encoder_input_chat_text' : encoder_inputs_chat_text, 'decoder_input_emotion' : emotion_inputs, 'decoder_input_prev_seq' : decoder_inputs, 'decoder_output' : decoder_outputs})
        
        return encoder_inputs_chat_text, emotion_inputs, decoder_inputs, decoder_outputs

    @staticmethod
    def pad_punctuation(text):
        return re.sub(f"([{string.punctuation}])", r" \1 ", text)

    @staticmethod
    def prepare_data(dataset_mame):
        df = DataManager.load_dataframe(dataset_mame)
        
        # Extract data from the DataFrame
        chat_text, text_response, emotion = DataManager.extract_data_from_dataframe(df)
        
        return chat_text, text_response, emotion
        