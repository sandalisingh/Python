import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
import graphviz
from enum import Enum
from States import EmotionStates, get_emotion_index, logging
from tensorflow.keras.layers import RepeatVector, Reshape, Flatten, Input, InputLayer, Embedding, Concatenate, Add, Attention
from tensorflow.keras.layers import MultiHeadAttention, LSTM, Dense, LayerNormalization, RepeatVector
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend
from keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
# from keras_vis import heatmap

class DialogueGenerator:

    #   INITIALIZATION

    def __init__(self):
        self.MAX_SEQ_LENGTH = 50  # Maximum sequence length for input and output
        self.VOCAB_SIZE = 10000   # Vocabulary size
        self.EMBEDDING_DIM = 300  # Embedding dimension
        self.HIDDEN_DIM = 512     # Hidden dimension for LSTM layers
        self.EMOTION_SIZE = 1
        self.MODEL_NAME = "dialogue_generator_model.h5"
        self.TOKENIZER_NAME = "tokenizer.pkl"
        self.MODEL = None
        self.TOKENIZER = None

        self.init_tokenizer()
        self.init_model()

    def init_tokenizer(self):
        try:
            with open(self.TOKENIZER_NAME, 'rb') as tokenizer_file:
                self.TOKENIZER = pickle.load(tokenizer_file)
            logging("info", "Tokenizer loaded.")
        except:
            self.TOKENIZER = None
   
    def init_model(self):
        try:
            self.MODEL = load_model(self.MODEL_NAME)
            logging("info", "Model loaded.")
        except Exception as e:
            logging("error", "Error loading model: "+str(e))
            self.define_model()
       
    #   TOKENIZER

    def create_tokenizer(self, chat_text, text_response):
        if self.TOKENIZER is None:
            # Concatenate chat_text and text_response
            all_texts = chat_text + text_response
            print("Texts concatenated.\n")
            
            # Create tokenizer and fit on all texts
            self.TOKENIZER = Tokenizer(num_words=self.VOCAB_SIZE - 3, oov_token='<OOV>')
            self.TOKENIZER.fit_on_texts(all_texts)
            self.TOKENIZER.word_index['<start>'] = self.TOKENIZER.num_words + 1
            self.TOKENIZER.word_index['<end>'] = self.TOKENIZER.num_words + 2
            
            # Manually add <start> and <end> to index_word
            self.TOKENIZER.index_word[self.TOKENIZER.word_index['<start>']] = '<start>'
            self.TOKENIZER.index_word[self.TOKENIZER.word_index['<end>']] = '<end>'
            
            logging("info", "Tokenizer created.")
            print("Tokenizer size = ", self.VOCAB_SIZE)

    def save_tokenizer(self):
        with open(self.TOKENIZER_NAME, 'wb') as tokenizer_file:
            pickle.dump(self.TOKENIZER, tokenizer_file)
        logging("info", "Tokenizer saved at "+self.TOKENIZER_NAME)
        return tokenizer_file
    
    #   DATA HANDLING

    def load_dataframe(self):
        df = pd.read_csv('Conversation3.csv')
        self.display_top_rows(df)
        self.zeroth_row = df.iloc[0]
        logging("info","Dataframe loaded.")
        return df
    
    def display_top_rows(self, df):
        print("Top 5 rows of the dataset:\n%s", df.head())
 
    def extract_data_from_dataframe(self, df):
        chat_text = df['chat_text'].tolist()
        text_response = df['text_response'].tolist()
        emotion = df['emotion'].tolist()
        logging("info","Data extracted from Dataframe.")
        return chat_text, text_response, emotion

    def preprocess_data(self, chat_text, text_response, emotion):
        print("\n\n-> PREPROCESS DATA")

        print("\nChat text[0] = ", chat_text[0])
        print("Text response[0] = ", text_response[0])
        print("Emotion[0] = ", emotion[0])

        chat_text_sequences = self.TOKENIZER.texts_to_sequences(chat_text)
        text_response_sequences = self.TOKENIZER.texts_to_sequences(text_response)
        print("\nConverted to indices...")
        print("Chat Text Sequence[0] = ", chat_text_sequences[0])
        print("Text Response Sequence[0] = ", text_response_sequences[0])

        # Add <end> token to each chat text sequence
        for seq in chat_text_sequences:
            seq.append(self.TOKENIZER.word_index['<end>'])
        
        # Add <start> token to each text response sequence
        for seq in text_response_sequences:
            seq.insert(0, self.TOKENIZER.word_index['<start>'])
        
        # Add <end> token to each text response sequence
        for seq in text_response_sequences:
            seq.append(self.TOKENIZER.word_index['<end>'])

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
        
        logging("info","Data preprocessed.")

        encoder_inputs_chat_text = tf.convert_to_tensor(encoder_inputs_chat_text, dtype=tf.float32)
        emotion_inputs = tf.convert_to_tensor(emotion_inputs, dtype=tf.float32)
        decoder_inputs = tf.convert_to_tensor(decoder_inputs, dtype=tf.float32)
        decoder_outputs = tf.convert_to_tensor(decoder_outputs, dtype=tf.float32)

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

    #   MODEL ARCHITECTURE 

    def generate_positional_encoding(self):
        angle_rads = self.get_angles(np.arange(self.MAX_SEQ_LENGTH)[:, np.newaxis],
                                     np.arange(self.EMBEDDING_DIM)[np.newaxis, :])
        
        print("Angle rads type:", type(angle_rads), "Shape:", angle_rads.shape)
        
        # Apply sin to even indices in the array
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]

        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        
        return pos_encoding
    
    def get_angles(self, position, i):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.EMBEDDING_DIM))
        return position * angle_rates

    def define_encoder(self):
        chat_text = Input(shape=(self.MAX_SEQ_LENGTH,), name='encoder_input_chat_text')
        embedding_output = Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM, mask_zero=True, name='dense_embedding_of_chat_text')(chat_text)
        pos_encoding = self.generate_positional_encoding()
        print("Pos encoding type:", type(pos_encoding))
        positional_output = Add(name='positional_embedding_of_chat_text')([embedding_output, pos_encoding])
        print("Positional output type before:", type(positional_output))
        attention_output = MultiHeadAttention(num_heads=8, key_dim=self.EMBEDDING_DIM, value_dim=self.EMBEDDING_DIM, name='multi_head_attention_to_chat_text')(positional_output, key=positional_output, value=positional_output)
        residual_output = Add(name='add_residual_connection_of_chat_text')([positional_output, attention_output])
        normalised_output = LayerNormalization(name='normalization_of_chat_text')(residual_output)
        lstm_output_seq, _, _ = LSTM(self.HIDDEN_DIM, return_sequences=True, return_state=True, name='lstm_of_chat_text')(normalised_output)
        return chat_text, lstm_output_seq

    def define_decoder(self, chat_text, encoder_output_seq):
        emotion = Input(shape=(1,), name='decoder_input1_emotion')
        prev_seq = Input(shape=(self.MAX_SEQ_LENGTH,), name='decoder_input2_prev_seq')
        embedding_output = Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM, mask_zero=True, name='dense_embedding_of_prev_seq')(prev_seq)
        # pos_encoding = self.generate_positional_encoding()
        # positional_output = Add(name='positional_embedding_of_decoder_2')([embedding_output, pos_encoding])
        projection_output = Dense(self.HIDDEN_DIM, name='projection_of_prev_seq')(embedding_output)
        attention_output = Attention(name='attention_to_prev_and_encoder_outputs')([projection_output, encoder_output_seq])
        residual_prev_seq_output = Add(name='add_residual_connection_of_prev_seq')([projection_output, attention_output])
        repeated_emotion = RepeatVector(self.MAX_SEQ_LENGTH, name='repeat_emotion')(emotion)
        reshaped_emotion = repeated_emotion * tf.cast(tf.ones((1, self.HIDDEN_DIM)), dtype=tf.float32)
        emotion_attention_output = Attention(name='attention_to_emotion')([residual_prev_seq_output, reshaped_emotion])
        residual_emotion_output = Add(name='add_residual_connection_of_emotion')([emotion_attention_output, reshaped_emotion])
        normalised_output = LayerNormalization(name='normalization_of_seq')(residual_emotion_output)
        dense_output = Dense(self.VOCAB_SIZE, name='dense_decoder_layer', activation='softmax')(normalised_output)
        return emotion, prev_seq, dense_output

    def define_model(self):
        if self.MODEL is None:
            chat_text, encoder_output_seq = self.define_encoder()
            emotion, prev_seq, output_seq = self.define_decoder(chat_text, encoder_output_seq) 
            self.MODEL = Model([chat_text, emotion, prev_seq], output_seq)
            
            logging("info","Model defined.")
            self.get_model_summary()

            # Plot the model graph with colors and save it
            # plot_model(self.model, to_file='model_graph_plot.png', show_shapes=True, show_layer_names=True)
            self.get_arch_flowchat()

    def save_model(self):
        if self.MODEL is not None:
            try:
                self.MODEL.save(os.path.join(os.getcwd(), self.MODEL_NAME))
                logging("info", "Model saved.")
            except Exception as e:
                logging("error", "Error saving model: " + str(e))
        else:
            logging("error", "Model not loaded, cannot save.")

    #   VISUALIZATION

    def get_model_summary(self):
        print("Model Summary:")
        self.MODEL.summary()

    def get_arch_flowchat(self):
        plot_model(self.MODEL, to_file='model_graph_plot.png', show_shapes=True)

    def generate_model_arch_plot(self, to_file):
        # Create a new graph
        dot = graphviz.Digraph()

        # Add nodes
        dot.node('InputLayer', color='lightblue')
        dot.node('Embedding', color='royalblue')
        dot.node('Add', color='lightcoral')
        dot.node('MultiHeadAttention', color='gold')
        dot.node('LayerNormalization', color='lightgreen')
        dot.node('LSTM', color='coral')
        dot.node('Dense', color='lightgreen')
        dot.node('RepeatVector', color='mediumpurple')
        dot.node('Attention', color='gold')

        # Add edges
        dot.edge('InputLayer', 'Embedding')
        dot.edge('Embedding', 'Add')
        dot.edge('Add', 'MultiHeadAttention')
        dot.edge('MultiHeadAttention', 'LayerNormalization')
        dot.edge('LayerNormalization', 'LSTM')
        dot.edge('LSTM', 'Dense')
        dot.edge('Dense', 'RepeatVector')
        dot.edge('RepeatVector', 'Attention')

        # Save the graph to a file or view directly
        if to_file:
            dot.render(to_file, format='png')
        else:
            dot.view()
    
    def visualize_tensor_value_range(self, layer_index, layer_name, tensor):
        try:  
            plt.figure(figsize=(10, 5))
            plt.plot(tensor)
            plt.title(f'Layer {layer_index} - {layer_name}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging("error",f"Layer {layer_index} - {layer_name}: Error occurred during visualization - {str(e)}")

    def visualize_attention_heatmap(self, attention_weights, layer_name, layer_index):
        attention_weights = attention_weights.squeeze()  # Remove unnecessary dimensions
        plt.figure(figsize=(10, 5))
        plt.title(f'Attention Heatmap - Layer {layer_index}: {layer_name}')
        plt.imshow(attention_weights, cmap='viridis')
        plt.xlabel('Input Sequence')
        plt.ylabel('Output Sequence')
        plt.colorbar()
        plt.show()

    #   OUTPUT INSPECTION

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

        # Create a model to extract intermediate outputs
        intermediate_model = Model(inputs=self.MODEL.inputs, outputs=[layer.output for layer in self.MODEL.layers])

        # Get intermediate outputs
        intermediate_results = intermediate_model.predict([chat_text, emotion, prev_seq])

        # Print intermediate output tensors along with the layer name and index for the first row
        for i, (layer, result) in enumerate(zip(self.MODEL.layers, intermediate_results)):
            layer_name = layer.name
            print(f"\nLayer {i} - '{layer_name}':")
            print(result[0])  # Printing only the first row of the tensor

            if isinstance(layer, Attention) and layer.get_weights():
                attention_weights = layer.get_weights()[0]  # Extract attention weights
                self.visualize_attention_heatmap(attention_weights, layer_name, i)

            # Visualize the tensor
            # self.visualize_tensor_value_range(i, layer_name, result[0])

    #   TRAINING

    def train_model(self, encoder_inputs, emotion_inputs, decoder_inputs, decoder_outputs, batch_size, epochs):
        self.MODEL.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        tensorboard = TensorBoard(log_dir='logs', histogram_freq=1)
        self.MODEL.fit([encoder_inputs, emotion_inputs, decoder_inputs], decoder_outputs, batch_size=batch_size, 
            epochs=epochs, validation_split=0.2, callbacks=[tensorboard])
        
        logging("info", "Model trained.")
 
    def create_train_and_save_model(self):
        chat_text, text_response, emotion = self.prepare_data()
        
        # Preprocess data
        encoder_inputs_chat_text, emotion_inputs, decoder_inputs, decoder_outputs = self.preprocess_data(chat_text, text_response, emotion)

        # Train the model
        self.train_model(encoder_inputs_chat_text, emotion_inputs, decoder_inputs, decoder_outputs, batch_size=64, epochs=1)           

        # Save the tokenizer
        self.save_tokenizer()

        self.save_model()

    #   GENERTE RESPONSE

    def sequence_to_text(self, prev_seq):
        # Convert the sequence of token indices to text
        response_tokens = [self.TOKENIZER.index_word.get(idx.numpy(), "<OOV>") for idx in tf.reshape(prev_seq, (-1,))]
        
        # Remove tokens after <end> token
        end_index = response_tokens.index('<end>') if '<end>' in response_tokens else len(response_tokens)
        response_text = ' '.join(response_tokens[:end_index])
        return response_text

    def generate_response_with_greedy_approach(self, chat_text_str, emotion_str):
        print("\n\n-> GENERATE RESPONSE")

        chat_text_input, emotion_input, _, prev_seq = self.preprocess_data([chat_text_str], [""], [emotion_str])

        for i in range(1, self.MAX_SEQ_LENGTH):
            # Predict the next token
            predictions = self.MODEL.predict([chat_text_input, emotion_input, prev_seq])
            
            # Get the token index with the highest probability (greedy approach)
            predicted_token_index = np.argmax(predictions[0, i - 1, :])
            
            # Update the previous sequence tensor
            updated_value = tf.constant(predicted_token_index, dtype=tf.int32)
            prev_seq = tf.tensor_scatter_nd_update(prev_seq, [[0, i]], [updated_value])

            # If the predicted token is an end token, break the loop
            if predicted_token_index == self.TOKENIZER.word_index['<end>']:
                break
            
        response_text = self.sequence_to_text(prev_seq)

        logging("info", "Response generated")
        
        return response_text 

    def generate_response_with_beam_search(self, chat_text_str, emotion_str, beam_width=5):
        # Preprocess data
        chat_text_input, emotion_input, _, prev_seq = self.preprocess_data([chat_text_str], [""], [emotion_str])

        # Initialize beam search
        beam = [(prev_seq, 0)]
        print("Beam = ", beam)

        # Initialize the final generated sequences
        final_sequences = []

        # Main loop for generating sequences
        for _ in range(self.MAX_SEQ_LENGTH):
            candidates = []
            for prev_seq, score in beam:
                # Predict the next token probabilities
                predictions = self.MODEL.predict([chat_text_input, emotion_input, prev_seq])

                # Get the top tokens with their probabilities
                top_tokens = np.argsort(predictions[0, -1])[-beam_width:]
                print("Top tokens = ", top_tokens)

                for token in top_tokens:
                    print("Token = ", token)

                    # Create a candidate sequence
                    candidate_seq = np.copy(prev_seq)

                    # Find the position where the next token should be inserted
                    next_token_position = np.where(candidate_seq == 0)[1][0]

                    # Insert the token at the correct position
                    candidate_seq[0, next_token_position] = token

                    # Calculate the score for the candidate sequence
                    candidate_score = score - np.log(predictions[0, -1, token])

                    if token is self.TOKENIZER.word_index['<end>']:
                        final_sequences.extend((candidate_seq, candidate_score))
                    else:
                        # Append the candidate to the list
                        candidates.append((candidate_seq, candidate_score))

            # Sort the candidates by score
            candidates.sort(key=lambda x: x[1])
            print("Candidates = ", candidates)

            # Select top candidates to continue beam search
            beam = candidates[:beam_width]
            print("Beam = ", beam)

            # Check for completion of sequences
            completed_sequences = [(seq, score) for seq, score in beam if seq[0, -1] == 9999]  # 2 is the end token
            if completed_sequences:
                print("Completed sequences = ", completed_sequences)
                final_sequences.extend(completed_sequences)

            # Filter out completed sequences from beam
            beam = [(seq, score) for seq, score in beam if seq[0, -1] != 9999]
            print("Beam = ", beam)
            print("Final sequences = ", final_sequences)

            # If no more candidates, break the loop
            if not beam:
                break

        # Choose the best final sequence
        if final_sequences:
            final_sequences.sort(key=lambda x: x[1])
            best_seq = final_sequences[0][0]

            # Extract and print all final sequences
            all_sequences = [self.sequence_to_text(seq[0]) for seq in final_sequences]
            print("All Final Sequences:")
            for seq in all_sequences:
                print(seq)
        else:
            # Choose the best sequence from the last beam
            beam.sort(key=lambda x: x[1])
            best_seq = beam[0][0]

        # Decode the best sequence into text
        response_text = self.sequence_to_text(best_seq)

        return response_text
