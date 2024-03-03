import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from enum import Enum
from States import EmotionStates, get_emotion_index
from tensorflow.keras.layers import RepeatVector, Reshape, Flatten, Input, InputLayer, Embedding, Concatenate, Add, Attention, MultiHeadAttention, LSTM, Dense, LayerNormalization, RepeatVector
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend
from tensorflow.keras.utils import plot_model

class DialogueGenerator:

    def __init__(self):
        self.MAX_SEQ_LENGTH = 50  # Maximum sequence length for input and output
        self.VOCAB_SIZE = 10000   # Vocabulary size
        self.EMBEDDING_DIM = 300  # Embedding dimension
        self.HIDDEN_DIM = 512     # Hidden dimension for LSTM layers
        self.EMOTION_SIZE = 1
        self.model = None
           
    def generate_positional_encoding(self):
        position_encoding = tf.expand_dims(tf.cast(tf.range(self.MAX_SEQ_LENGTH), dtype=tf.float32), axis=-1) / tf.cast((self.EMBEDDING_DIM - 1), dtype=tf.float32)
        position_encoding = position_encoding * tf.cast(tf.ones((1, self.EMBEDDING_DIM)), dtype=tf.float32)
        position_encoding = tf.expand_dims(position_encoding, 0)
        return position_encoding

    #   TOKENIZER

    def create_tokenizer(self, chat_text, text_response):
        # Concatenate chat_text and text_response
        all_texts = chat_text + text_response
        print("Texts concatenated.\n")
        
        # Create tokenizer and fit on all texts
        self.tokenizer = Tokenizer(num_words=self.VOCAB_SIZE - 3, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(all_texts)
        self.tokenizer.word_index['<start>'] = self.tokenizer.num_words + 1
        self.tokenizer.word_index['<end>'] = self.tokenizer.num_words + 2
        
        # Manually add <start> and <end> to index_word
        self.tokenizer.index_word[self.tokenizer.word_index['<start>']] = '<start>'
        self.tokenizer.index_word[self.tokenizer.word_index['<end>']] = '<end>'
        
        print("Tokenizer created and fitted on texts.")
        print("Tokenizer size = ", self.VOCAB_SIZE)

    def save_tokenizer(self):
        tokenizer_path = "tokenizer.pkl"
        with open(tokenizer_path, 'wb') as tokenizer_file:
            pickle.dump(self.tokenizer, tokenizer_file)
        print("\nTokenizer saved at ", tokenizer_path)
        return tokenizer_file
    
    #   DATA HANDLING

    def load_dataframe(self):
        df = pd.read_csv('Conversation3.csv')
        self.display_top_rows(df)
        self.zeroth_row = df.iloc[0]
        return df
    
    def display_top_rows(self, df):
        print("Top 5 rows of the dataset:\n%s", df.head())
 
    def extract_data_from_dataframe(self, df):
        chat_text = df['chat_text'].tolist()
        text_response = df['text_response'].tolist()
        emotion = df['emotion'].tolist()
        print("\nData extracted from DataFrame.")
        return chat_text, text_response, emotion

    def preprocess_data(self, chat_text, text_response, emotion):
        print("\n\n-> PREPROCESS DATA")

        print("\nChat text[0] = ", chat_text[0])
        print("Text response[0] = ", text_response[0])
        print("Emotion[0] = ", emotion[0])

        chat_text_sequences = self.tokenizer.texts_to_sequences(chat_text)
        text_response_sequences = self.tokenizer.texts_to_sequences(text_response)
        print("\nConverted to indices...")
        print("Chat Text Sequence[0] = ", chat_text_sequences[0])
        print("Text Response Sequence[0] = ", text_response_sequences[0])

        # Add <end> token to each chat text sequence
        for seq in chat_text_sequences:
            seq.append(self.tokenizer.word_index['<end>'])
        
        # Add <start> token to each text response sequence
        for seq in text_response_sequences:
            seq.insert(0, self.tokenizer.word_index['<start>'])
        
        # Add <end> token to each text response sequence
        for seq in text_response_sequences:
            seq.append(self.tokenizer.word_index['<end>'])

        print("\nAdded <end> token...")
        print("Chat Text Sequence[0] = ", chat_text_sequences[0])
        print("Text Response Sequence[0] = ", text_response_sequences[0])
        
        # Pad sequences
        encoder_inputs_chat_text = tf.keras.preprocessing.sequence.pad_sequences(chat_text_sequences, maxlen=self.MAX_SEQ_LENGTH, padding='post')
        decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(text_response_sequences, maxlen=self.MAX_SEQ_LENGTH, padding='post')
        print("\nPadding...")
        print("Encoder input[0] = ", encoder_inputs_chat_text[0])
        print("Decoder input[0] = ", decoder_inputs[0])

        # Shift targets for teacher forcing
        decoder_outputs = np.zeros_like(decoder_inputs)
        decoder_outputs[:, :-1] = decoder_inputs[:, 1:]                                    # ??? why shifting
        decoder_outputs[:, -1] = 0
        print("\nDecoder output[0] = ", decoder_outputs[0])

        # Map emotion strings to their corresponding enum values
        emotion_inputs = [get_emotion_index(emo) for emo in emotion] 
        print("\nemotion_index[0] = ", emotion_inputs[0])

        # convert list to vector
        emotion_inputs = np.array(emotion_inputs)
        emotion_inputs = np.expand_dims(emotion_inputs, axis=1)
        
        print("Data preprocessed.\n\n")

        encoder_inputs_chat_text = tf.convert_to_tensor(encoder_inputs_chat_text)
        emotion_inputs = tf.convert_to_tensor(emotion_inputs)
        decoder_inputs = tf.convert_to_tensor(decoder_inputs)
        decoder_outputs = tf.convert_to_tensor(decoder_outputs)

        print("\nAfter Preprocess...")
        print("encoder_inputs_chat_text = ", encoder_inputs_chat_text)
        print("emotion_inputs = ", emotion_inputs)
        print("decoder_inputs = ", decoder_inputs)
        print("decoder_outputs = ", decoder_outputs)
        
        return encoder_inputs_chat_text, emotion_inputs, decoder_inputs, decoder_outputs

    def prepare_data(self):
        df = self.load_dataframe()
        
        # Extract data from the DataFrame
        chat_text, text_response, emotion = self.extract_data_from_dataframe(df)
        
        self.create_tokenizer(chat_text, text_response)
        return chat_text, text_response, emotion

    #   MODEL ARCHITECTURE DEFINITION

    def define_encoder(self):
        print("\n\n-> ENCODER")

        print("\n- LAYER 0 - INPUT")
        chat_text = Input(shape=(self.MAX_SEQ_LENGTH,), name='encoder_input_chat_text')
        print("Encoder Chat Text Input: ", chat_text)
        
        print("\n- LAYER 1 - EMBEDDING")
        embedding_output = Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM, mask_zero=True, name='dense_embedding_of_chat_text')(chat_text)
        print("Encoder Embedding Output: ", embedding_output)

        print("\n- LAYER 2 - (ORDER) - POSITIONAL EMBEDDING")
        positional_output = Add(name='positional_embedding_of_chat_text')([embedding_output, self.generate_positional_encoding()])
        print("Positional Embeddings Output: ", positional_output)

        print("\n- LAYER 3 - (CONTEXT) - MULTI HEAD ATTENTION")
        attention_output = MultiHeadAttention(num_heads=8, key_dim=self.EMBEDDING_DIM, value_dim=self.EMBEDDING_DIM, name='multi_head_attention_to_chat_text')(positional_output, key=positional_output, value=positional_output)
        print("Attention Output: ", attention_output)

        print("\n- LAYER 4 - ADDING RESIDUAL CONNECTION")
        residual_output = Add(name='add_residual_connection_of_chat_text')([positional_output, attention_output])
        print("Residual Addition: ", residual_output)

        print("\n- LAYER 5 - NORMALIZATION")
        normalised_output = LayerNormalization(name='normalization_of_chat_text')(residual_output)
        print("Normalization: ", normalised_output)

        print("\n- LAYER 6 - (HISTORY) - LSTM")
        lstm_output_seq, _, _ = LSTM(self.HIDDEN_DIM, return_sequences=True, return_state=True, name='lstm_of_chat_text')(normalised_output)
        print("LSTM Seq: ", lstm_output_seq)

        print("\nEncoder defined.")

        return chat_text, lstm_output_seq

    def define_decoder(self, chat_text, encoder_output_seq):
        print("\n\n-> DECODER")

        print("\n- LAYER 0 - EMOTION INPUT")
        emotion = Input(shape=(1,), name='decoder_input1_emotion')
        print("Decoder Emotion Input: ", emotion)
        
        print("\n- LAYER 1 - DECODER PREV SEQ INPUT")
        prev_seq = Input(shape=(self.MAX_SEQ_LENGTH,), name='decoder_input2_prev_seq')
        print("Decoder Prev Seq Input: ", prev_seq)

        print("\n- LAYER 2 - EMBEDDING OF PREV SEQ")
        embedding_output = Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM, mask_zero=True, name='dense_embedding_of_prev_seq')(prev_seq)
        print("Decoder Embedding Output: ", embedding_output)

        print("\n- LAYER 3 - (ORDER) - POSITIONAL EMBEDDING")
        positional_output = Add(name='positional_embedding_of_decoder_2')([embedding_output, self.generate_positional_encoding()])
        print("Positional Embeddings of Decoder Output: ", positional_output)

        print("\n- LAYER 4 - PROJECTION OF PREV SEQ")
        projection_output = Dense(self.HIDDEN_DIM, name='projection_of_prev_seq')(positional_output)
        print("Projection: ", projection_output)
       
        print("\n- LAYER 5 - ATTENTION")
        attention_output = Attention(name='attention_to_prev_and_encoder_outputs')([projection_output, encoder_output_seq])
        print("Attention Output: ", attention_output)

        print("\n- LAYER 6 - REPEAT EMOTION")
        repeated_emotion = RepeatVector(self.MAX_SEQ_LENGTH, name='repeat_emotion')(emotion)
        print("Repeated emotion: ", repeated_emotion)

        print("\n- LAYER 7 - RESHAPE EMOTION")
        reshaped_emotion = repeated_emotion * tf.cast(tf.ones((1, self.HIDDEN_DIM)), dtype=tf.float32)
        print("Reshaped emotion: ", reshaped_emotion)

        print("\n- LAYER 8 - ATTENTION")
        emotion_attention_output = Attention(name='attention_to_emotion')([attention_output, reshaped_emotion])
        print("Attention Output: ", attention_output)
        
        print("\n- LAYER 9 - DENSE")
        dense_output = Dense(self.VOCAB_SIZE, name='dense_decoder_layer', activation='softmax')(emotion_attention_output)
        print("Dense Output: ", dense_output)
        
        print("\nDecoder defined.\n")
        
        return emotion, prev_seq, dense_output

    def define_model(self):
        if self.model is None:
            chat_text, encoder_output_seq = self.define_encoder()
            emotion, prev_seq, output_seq = self.define_decoder(chat_text, encoder_output_seq) 

            print("Defining model...")
            print("chat_text shape = ", chat_text.shape)
            print("encoder_output_seq shape = ", encoder_output_seq.shape)
            print("emotion shape = ", emotion.shape)
            print("prev_seq shape = ", prev_seq.shape)
            print("output_seq shape = ", output_seq.shape)

            self.model = Model([chat_text, emotion, prev_seq], output_seq)
            
            print("\nModel defined.")
            print("Model Summary:")
            self.model.summary()

            # Plot the model graph with colors and save it
            plot_model(self.model, to_file='model_graph_plot.png', show_shapes=True, show_layer_names=True)

    #   OUTPUT INSPECTION

    def visualize_tensor(self, layer_index, layer_name, tensor):
        try:  
            plt.figure(figsize=(10, 5))
            plt.plot(tensor[0])
            plt.title(f'Layer {layer_index} - {layer_name}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Layer {layer_index} - {layer_name}: Error occurred during visualization - {str(e)}")

    def inspect_layer_outputs(self):
        chat_text, text_response, emotion = self.prepare_data()
        
        # Enable eager execution
        tf.config.experimental_run_functions_eagerly(True)  

        # Randomly select an index
        random_index = np.random.randint(0, len(chat_text))

        chat_text = [chat_text[random_index]]
        emotion = [emotion[random_index]]
        text_response = [text_response[random_index]]

        print("\nINPUT:\n-> chat text = ", chat_text)
        print("-> emotion = ", emotion)
        print("-> text response = ", text_response)

        chat_text, emotion, prev_seq, _ = self.preprocess_data(chat_text, text_response, emotion)   

        self.define_model()

        # Create a model to extract intermediate outputs
        intermediate_model = Model(inputs=self.model.inputs, outputs=[layer.output for layer in self.model.layers])

        # Get intermediate outputs
        intermediate_results = intermediate_model.predict([chat_text, emotion, prev_seq])

        # Print intermediate output tensors along with the layer name and index for the first row
        for i, (layer, result) in enumerate(zip(self.model.layers, intermediate_results)):
            layer_name = layer.name
            print(f"\nLayer {i} - '{layer_name}':")
            print(result[0])  # Printing only the first row of the tensor

            # Visualize the tensor
            self.visualize_tensor(i, layer_name, result)

    #   TRAINING

    def train_model(self, encoder_inputs, emotion_inputs, decoder_inputs, decoder_outputs, batch_size, epochs):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        self.model.fit([encoder_inputs, emotion_inputs, decoder_inputs], decoder_outputs, batch_size=batch_size, 
            epochs=epochs, validation_split=0.2)
        
        print("\nModel trained.\n")
 
    def create_train_and_save_model(self):
        chat_text, text_response, emotion = self.prepare_data()
        
        # Preprocess data
        encoder_inputs_chat_text, emotion_inputs, decoder_inputs, decoder_outputs = self.preprocess_data(chat_text, text_response, emotion)

        # Define and compile the model
        self.define_model()

        # Train the model
        self.train_model(encoder_inputs_chat_text, emotion_inputs, decoder_inputs, decoder_outputs, batch_size=64, epochs=1)           

        # Save the tokenizer
        self.save_tokenizer()

        # Save the trained model
        self.model.save("dialogue_generator_model.keras")
        print("Trained model saved.\n")

    #   GENERTE RESPONSE

    def generate_response_with_greedy_approach(self, chat_text, emotion_str, max_seq_length):
        print("\n\n-> GENERATE RESPONSE")

        # Preprocess input text
        chat_text_sequence = self.tokenizer.texts_to_sequences([chat_text])
        print("Chat text sequence = ", chat_text_sequence)
        chat_text_sequence = tf.keras.preprocessing.sequence.pad_sequences(chat_text_sequence, maxlen=max_seq_length, padding='post')
        print("Padded chat text sequence = ", chat_text_sequence)

        # Convert emotion string to its corresponding enum value
        emotion = EmotionStates[emotion_str.strip().replace(' ', '_').title()].value
        print("Emotion = ", emotion)
        
        # Preprocess emotion data
        emotion_sequence = np.zeros((1, self.EMOTION_SIZE))
        emotion_sequence[0, emotion] = 1  # Set the corresponding emotion index to 1
        print("Emotion sequence = ", emotion_sequence)

        # Initialize conversation history with encoder input and emotion
        conversation_history = np.concatenate([chat_text_sequence, emotion_sequence], axis=1)
        print("Conversational history = ", conversation_history)
        
        # Initialize decoder input with a start token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.tokenizer.word_index['<start>']
        print("Target sequence = ", target_seq)
        
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens = self.model.predict([conversation_history, target_seq])
            print("\n\nOutput tokens[0] = ", output_tokens[0])

            # greedy decoding -  selecting the token with the highest probability
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            print("Sampled token index = ", sampled_token_index)

            # Get the indices of the top three probabilities
            top_three_indices = np.argsort(output_tokens[0, -1, :])[::-1][:3]

            # Initialize a list to store the top three tokens
            top_three_tokens = []

            # Iterate over the top three indices and get the corresponding tokens
            for index in top_three_indices:
                token = self.tokenizer.index_word.get(index, None)
                if token is not None:
                    top_three_tokens.append((token, output_tokens[0, -1, index]))

            # Print the top three tokens and their probabilities
            print("\nTop three tokens with highest probabilities:")
            for token, probability in top_three_tokens:
                print(f"Token: {token}, Probability: {probability}")

            sampled_word = self.tokenizer.index_word.get(sampled_token_index, None)
            print("\nSampled word = ", sampled_word)

            if sampled_word is not None and sampled_word != '<end>':
                decoded_sentence += ' ' + sampled_word
                print("Decoded sentence = ", decoded_sentence)

            if sampled_word == '<end>' or len(decoded_sentence.split()) > max_seq_length:
                stop_condition = True
                print("\n!! Stopping condition achieved\n")

            target_seq = np.zeros((1, 1))  # Resetting target_seq for next iteration
            target_seq[0, 0] = sampled_token_index
            print("Target sequence = ", target_seq)
            
            # Update conversation history with the latest decoder output
            conversation_history = np.concatenate([conversation_history, target_seq], axis=1)
            print("Conversation history = ", conversation_history)
        
        print("\nResponse generated.\n")
        
        return decoded_sentence.strip()

    def generate_response_with_beam_search(self, chat_text, emotion_str, max_seq_length, beam_width=3, length_penalty_factor=0.6):
        # Preprocess input text
        chat_text_sequence = self.tokenizer.texts_to_sequences([chat_text])
        chat_text_sequence = tf.keras.preprocessing.sequence.pad_sequences(chat_text_sequence, maxlen=max_seq_length, padding='post')
        
        # Convert emotion string to its corresponding enum value
        emotion = EmotionStates[emotion_str.strip().replace(' ', '_').title()].value
        
        # Preprocess emotion data
        emotion_sequence = np.zeros((1, self.EMOTION_SIZE))
        emotion_sequence[0, emotion] = 1  # Set the corresponding emotion index to 1
        
        # Initialize conversation history with encoder input and emotion
        conversation_history = np.concatenate([chat_text_sequence, emotion_sequence], axis=1)
        
        # Initialize decoder input with a start token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.tokenizer.word_index['<start>']
        
        # Initialize list to store beam search candidates
        candidates = [(target_seq, 0)]  # (candidate sequence, score)
        
        # Beam search loop
        for _ in range(max_seq_length):
            next_candidates = []
            
            # Flag to indicate if any candidate sequence ends with <end>
            end_found = False
            
            # Expand each candidate sequence
            for seq, score in candidates:
                if seq[0, -1] == self.tokenizer.word_index['<end>']:
                    # If a candidate sequence ends with <end>, mark the flag and skip expansion
                    end_found = True
                    continue
                
                output_tokens = self.model.predict([conversation_history, seq])
                sampled_token_probs = output_tokens[0, -1, :]
                
                # Get top beam_width tokens
                top_indices = np.argsort(sampled_token_probs)[-beam_width:]
                
                # Expand each candidate with top beam_width tokens
                for index in top_indices:
                    next_seq = np.concatenate([seq, np.zeros((1, 1))], axis=1)
                    next_seq[0, -1] = index
                    
                    # Calculate score with length normalization
                    next_score = score - np.log(sampled_token_probs[index])  # Length normalization
                    next_candidates.append((next_seq, next_score))
            
            if end_found:
                break  # If any candidate sequence ends with <end>, terminate the search
            
            # Select top beam_width candidates
            candidates = sorted(next_candidates, key=lambda x: x[1])[:beam_width]

        # Select the best candidate sequence
        best_seq, _ = candidates[0]
        
        # Convert token indices to words
        decoded_sentence = ' '.join(self.tokenizer.index_word.get(idx, '<OOV>') for idx in best_seq[0])
        
        return decoded_sentence.strip()
