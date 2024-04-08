import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from States import logging, EmotionStates
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import backend
from PositionalEncoding import PositionalEncoding
from DataManager import DataManager
from DataVisualizer import DataVisualizer
from tensorflow.keras import regularizers
from Tokenizer import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import string

class DialogueGenerator:

    #   INITIALIZATION

    def __init__(self):
        self.MAX_SEQ_LENGTH = 50  # Maximum sequence length for input and output
        self.VOCAB_SIZE = 10000   # Vocabulary size
        self.EMBEDDING_DIM = 300  # Embedding dimension
        self.HIDDEN_DIM = 512     # Hidden dimension for LSTM layers
        self.MODEL_NAME = "dialogue_generator_model"
        self.MODEL = None
        self.TOKENIZER = Tokenizer(self.VOCAB_SIZE)

        self.load_model()

    def load_model(self):
        try:
            self.MODEL = load_model(self.MODEL_NAME+".keras")
            logging("info", "Model loaded.")
        except Exception as e:
            logging("error", "Error loading model: "+str(e))
            self.define_model()
    
    def save_model(self):
        if self.MODEL is not None:
            try:
                self.MODEL.save(os.path.join(os.getcwd(), self.MODEL_NAME+".keras"))
                logging("info", "Model saved.")
            except Exception as e:
                logging("error", "Error saving model: " + str(e))
        else:
            logging("error", "Model not loaded, cannot save.")
       
    #   MODEL ARCHITECTURE 

    def define_encoder(self):
        chat_text = Input(shape=(self.MAX_SEQ_LENGTH,), name='chat_text_input')
        embedding_output = Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM, mask_zero=True, name='embedding_of_chat_text')(chat_text)
        
        positional_layer = PositionalEncoding(self.MAX_SEQ_LENGTH, self.EMBEDDING_DIM, name='positional_encoding_of_chat_text')
        positional_output = positional_layer(embedding_output)
        # positional_output = Dropout(0.2)(positional_output)
        
        attention_output = MultiHeadAttention(num_heads=8, key_dim=self.EMBEDDING_DIM, value_dim=self.EMBEDDING_DIM, name='self_attention_to_chat_text')(positional_output, positional_output)
        residual_output = Add(name='residual_connection_of_chat_text')([positional_output, attention_output])
        residual_output = Dropout(0.2)(residual_output) 
        normalised_output = LayerNormalization(name='normalization_of_chat_text')(residual_output)
        _, hidden_state, cell_state = LSTM(self.HIDDEN_DIM, return_sequences=False, return_state=True, name='summarization_of_chat_text')(normalised_output)

        return chat_text, normalised_output, hidden_state, cell_state

    def define_decoder(self, processed_chat_text, summary_hidden_state, summary_cell_state):
        # Handle previous sequence
        prev_seq = Input(shape=(self.MAX_SEQ_LENGTH,), name='prev_seq_input') # bs,50
        embedding_output = Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM, mask_zero=True, name='embedding_of_prev_seq')(prev_seq) # bs,50,300
        
        positional_layer = PositionalEncoding(self.MAX_SEQ_LENGTH, self.EMBEDDING_DIM, name='positional_encoding_of_prev_seq') # bs,50,300
        positional_output = positional_layer(embedding_output) 
        # positional_output = Dropout(0.2)(positional_output)

        attention_output = Attention(name='self_attention_to_prev_seq')([positional_output, positional_output]) # bs,50,300
        residual_of_prev_seq = Add(name='residual_connection_of_prev_seq')([positional_output, attention_output]) # bs,50,300

        # Multiple lstm
        _, lstm_chat_text_hidden_state, lstm_chat_text_cell_state = LSTM(self.HIDDEN_DIM, return_sequences=False, return_state=True, name='lstm_of_chat_text')(processed_chat_text, initial_state=[summary_hidden_state, summary_cell_state])
        lstm_seq = LSTM(self.HIDDEN_DIM, return_sequences=True, return_state=False, name='lstm_of_prev_seq')(residual_of_prev_seq, initial_state=[lstm_chat_text_hidden_state, lstm_chat_text_cell_state]) # bs,50,512

        # Handle emotion
        emotion = Input(shape=(1,), name='emotion_input') # bs,1
        emotion_repeat = RepeatVector(self.MAX_SEQ_LENGTH, name='emotion_repeat')(emotion) # bs,50,1
        reshaped_emotion = Reshape((self.MAX_SEQ_LENGTH,), name='emotion_reshaped')(emotion_repeat) # bs,50
        emotion_embedding = Embedding(len(EmotionStates), self.HIDDEN_DIM, name='emotion_embedding')(reshaped_emotion)  # bs,50,512
        
        # Apply attention with emotion and LSTM output
        # context_vector = Attention(name='attention_to_emotion')([lstm_seq, emotion_embedding]) # bs,50,512

        # Concatenate context vector with LSTM output
        concatenated_output = Concatenate(name='seq_and_emotion')([lstm_seq, emotion_embedding]) # bs,50,1024
        concatenated_output = Dropout(0.2)(concatenated_output)
        dense_of_seq = Dense(self.VOCAB_SIZE, name='dense_of_seq', activation='softmax')(concatenated_output)

        return emotion, prev_seq, dense_of_seq

    def define_model(self):
        if self.MODEL is None:
            chat_text, processed_chat_text, summary_hidden_state, summary_cell_state = self.define_encoder()
            emotion, prev_seq, dense_of_seq = self.define_decoder(processed_chat_text, summary_hidden_state, summary_cell_state) 
            self.MODEL = Model([chat_text, emotion, prev_seq], dense_of_seq)

            # embedding_layer = self.MODEL.layers[1]  # Access the Embedding layer for L2 regularization
            # embedding_layer.kernel_regularizer = regularizers.l2(0.01)  

            # Emotion embedding with L2 regularization
            emotion_embedding_layer = self.MODEL.layers[-2]  # Access the emotion embedding layer
            emotion_embedding_layer.kernel_regularizer = regularizers.l2(0.01)

            self.model_visualization()
            
            logging("info","Model defined.")

            self.save_model()

    def reset_states(self):
        if self.MODEL is None:
            for layer in self.MODEL.layers:
                if isinstance(layer, tf.keras.layers.LSTM):
                    layer.reset_states()
    
    #   INSPECTION

    def model_visualization(self):
        DataVisualizer.get_model_summary(self.MODEL)
        DataVisualizer.get_arch_flowchat(self.MODEL, "Plots/model_graph_plot.png")

    def inspect_layer_outputs(self, dataset_filename):
        chat_text, text_response, emotion = DataManager.prepare_data(dataset_filename)
        
        # Enable eager execution
        tf.config.run_functions_eagerly(True)  

        # Randomly select an index
        random_index = np.random.randint(0, len(chat_text))

        chat_text = [chat_text[random_index]]
        emotion = [emotion[random_index]]
        text_response = [text_response[random_index]]

        DataVisualizer.display_top_rows(pd.DataFrame({"chat_text": chat_text, "emotion": emotion, "text_response": text_response}), 1, "Input")

        chat_text, emotion, prev_seq, _ = DataManager.preprocess_data(chat_text, text_response, emotion, self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)   

        # Create a model to extract intermediate outputs
        intermediate_model = Model(inputs=self.MODEL.inputs, outputs=[layer.output for layer in self.MODEL.layers])

        # Get intermediate outputs
        intermediate_results = intermediate_model.predict([chat_text, emotion, prev_seq])

        # Print intermediate output tensors along with the layer name and index for the first row
        for i, (layer, result) in enumerate(zip(self.MODEL.layers, intermediate_results)):
            layer_name = layer.name
            # print(f"\nLayer {i} - '{layer_name}':")
            # print(result[0])  # Printing only the first row of the tensor
            DataVisualizer.print_tensor_dict(f"Layer {i} - {layer_name}", {"Output tensor":result})

            # if isinstance(layer, Attention) and layer.get_weights():
            #     attention_weights = layer.get_weights()[0]  # Extract attention weights
            #     DataVisualizer.visualize_attention_heatmap(attention_weights, layer_name, i)

            # Visualize the tensor
            # DataVisualizer.visualize_tensor_value_range(i, layer_name, result[0])

    #   TRAINING
    
    def test_model(self, chat_text, text_response, emotion):
        # Preprocess data
        chat_text_input, emotion_input, prev_seq, output_seq = DataManager.preprocess_data(chat_text, text_response, emotion, self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)

        self.reset_states()

        loss, accuracy = self.MODEL.evaluate([chat_text_input, emotion_input, prev_seq], output_seq)
        logging("info", f"Model Tested: \t[Loss={loss}, Accuracy={accuracy}]")

    def train_model(self, chat_text, text_response, emotion, epochs):
        # Preprocess data
        chat_text_input, emotion_input, prev_seq, output_seq = DataManager.preprocess_data(chat_text, text_response, emotion, self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)

        self.reset_states()

        # Train the model
        self.MODEL.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
        
        # Define validation split for evaluation during training
        validation_split = 0.2  # 20% of training data for validation

        # Define early stopping based on validation loss and validation accuracy
        early_stopping_loss = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

        # Train the model with validation split
        history = self.MODEL.fit([chat_text_input, emotion_input, prev_seq], output_seq, batch_size=64, epochs=epochs, validation_split=validation_split, callbacks=[early_stopping_loss])

        DataVisualizer.plot_train_history(history.history, 'loss', 'accuracy', 'Model_Training')
        DataVisualizer.plot_train_history(history.history, 'val_loss', 'val_accuracy', 'Model_Validation')

        logging("info", "Model trained.")    

        self.save_model()

    def train_and_test(self, dataset_filename, epochs=10, test_size=0.2, random_state=42):
        chat_text, text_response, emotion = DataManager.prepare_data(dataset_filename)

        chat_train, chat_test, response_train, response_test, emotion_train, emotion_test = train_test_split(chat_text, text_response, emotion, test_size=test_size, random_state=random_state, shuffle=False)

        self.train_model(chat_train, response_train, emotion_train, epochs)
        self.test_model(chat_test, response_test, emotion_test)

        self.save_model()

    #   GENERTE RESPONSE

    def sequence_to_text(self, seq):
        seq = tf.reshape(seq[:,:,:], (-1,))
        seq = seq[1:]

        # Remove <OOV> and <start>
        # seq = [x for x in seq if (x!=1 and x!=9998)]

        # Convert the sequence of token indices to text
        response_tokens = [self.TOKENIZER.TOKENIZER.index_word.get(idx.numpy(), "") for idx in seq]
        # response_tokens = [token for token in response_tokens if token!=""]

        # Remove tokens after <end> token
        end_index = response_tokens.index('<end>') if '<end>' in response_tokens else len(response_tokens)
        # response_text = ' '.join(response_tokens[:end_index])
        response_text = ''
        for i in range(end_index):
            token = response_tokens[i]
            if token in string.punctuation:
                response_text += token
            else:
                response_text += ' ' + token
        response_text = response_text.strip()
        # response_text = ' '.join(response_tokens)
        return response_text

    def generate_response_with_greedy_approach(self, chat_text_str, emotion_str):
        chat_text_input, emotion_input, prev_seq, _ = DataManager.preprocess_data([chat_text_str], [""], [emotion_str], self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)
        prev_seq = tf.tensor_scatter_nd_update(prev_seq, indices=[[0, 1]], updates=[0])

        # DataVisualizer.print_tensor_dict("Input to model", {"chat_text": chat_text_input, "emotion": emotion_input, "prev_seq": prev_seq})

        for i in range(1, self.MAX_SEQ_LENGTH):
            # Predict the next token
            predicted_seq = self.MODEL.predict([chat_text_input, emotion_input, prev_seq], verbose=0)

            # Get the token index with the highest probability (greedy approach)
            predicted_token_index = np.argmax(predicted_seq[0, i-1, :])

            if predicted_token_index==1 or predicted_token_index==0:    # skip <OOV>
                continue
            
            # Update the previous sequence tensor
            updated_value = tf.constant(predicted_token_index, dtype=tf.int32)
            prev_seq = tf.tensor_scatter_nd_update(prev_seq, [[0, i]], [updated_value])

            # If the predicted token is an end token, break the loop
            if predicted_token_index == self.TOKENIZER.END_TOKEN:
                break
            
        response_text = self.sequence_to_text(tf.expand_dims(prev_seq, axis=0))

        logging("info", "Response generated")
        
        return response_text 

    def generate_response_with_beam_search(self, chat_text_str, emotion_str, beam_width=5):
        # Preprocess data 
        chat_text_input, emotion_input, prev_seq, _ = DataManager.preprocess_data([chat_text_str], [""], [emotion_str], self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)
        prev_seq = tf.tensor_scatter_nd_update(prev_seq, indices=[[0, 1]], updates=[0])

        # DataVisualizer.print_tensor_dict("Input to model", {"chat_text": chat_text_input, "emotion": emotion_input, "prev_seq": prev_seq})

        # Initialize beam search set
        beam_list = [(prev_seq, 1.0)]

        # Initialize the final generated sequences
        final_sequences_list = []

        # Main loop for generating sequences
        for i in range(1, self.MAX_SEQ_LENGTH):
            candidates_list = []
            completed_sequences_list = []
            for prev_seq, score in beam_list:
                # Predict the next token probabilities
                predicted_seq = self.MODEL.predict([chat_text_input, emotion_input, prev_seq], verbose=0)

                # Get the top tokens with their probabilities
                top_token_indices = np.argsort(predicted_seq[0, i-1, :])[-beam_width:]

                for token_index in top_token_indices:
                    # Create a candidate sequence
                    candidate_seq = np.copy(prev_seq)

                    # # Find the position where the next token should be inserted
                    # next_token_position = np.where(candidate_seq == 0)[1][0]

                    # # Insert the token at the correct position
                    # candidate_seq[0, next_token_position] = token_index

                    candidate_seq[0, i] = token_index

                    candidate_score = self.calculate_score(candidate_seq[0], chat_text_input[0].numpy(), i, score)

                    # Check if the sequence is complete
                    if token_index == self.TOKENIZER.END_TOKEN:
                        completed_sequences_list.append((candidate_seq, candidate_score))
                    else:
                        candidates_list.append((candidate_seq, candidate_score))

            # Select candidates with highest scores - sorted in asc order
            beam_list = sorted(candidates_list, key=lambda x: x[1])[-beam_width:]

            # Check for completion of sequences
            if len(completed_sequences_list)>0:
                final_sequences_list.append(completed_sequences_list)

            if final_sequences_list:
                final_sequences_list = sorted(final_sequences_list, key=lambda x: x[1]) # asc order
                best_seq = final_sequences_list[-1][0]   # highest score is the last row
            else:
                best_seq = beam_list[-1][0]  # highest score is the last row

        # Decode the best sequence into text
        response_text = self.sequence_to_text(np.array([best_seq]))

        return response_text

    def calculate_score(self, gen_sequence, input_text, seq_len, prev_score, diversity_weight=0.4, length_penalty_weight=0.4, responsiveness_weight=0.2, trigram_penalty=0.6):
        # Calculate diversity score
        unique_tokens = len(set(gen_sequence))
        n_grams = len(set(zip(*[gen_sequence[i:] for i in range(3)])))  # Consider trigrams for diversity
        diversity_score = (unique_tokens + n_grams) / (seq_len+1)  # Normalize diversity score

        # Calculate length penalty
        if seq_len < 4:
            length_penalty = seq_len / 4
        elif seq_len > 10:
            length_penalty = 10 / (seq_len*seq_len)
        else:
            length_penalty = 1
        
        # Calculate responsiveness
        responsiveness = 0
        input_tokens = set(input_text)
        generated_tokens = set(gen_sequence)
        common_tokens = len(input_tokens.intersection(generated_tokens))
        responsiveness = common_tokens / max(len(input_tokens), 1)
        
        # Normalize scores
        diversity_score = round(diversity_score, 4)
        length_penalty = round(length_penalty, 4)
        responsiveness = round(responsiveness, 4)
        
        final_score = (diversity_weight*diversity_score + length_penalty_weight*length_penalty + responsiveness_weight*responsiveness)
        
        # Ensure final score is between 0 and 1
        final_score = max(0, min(final_score, 1))
        
        # Normalize score considering previous score
        normalized_score = final_score + (prev_score - final_score) * 0.1
        
        # Round normalized score to 4 decimal places
        normalized_score = round(normalized_score, 4)
        
        return normalized_score
