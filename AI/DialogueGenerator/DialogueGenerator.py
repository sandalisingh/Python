import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from States import logging, EmotionStates
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend
from PositionalEncoding import PositionalEncoding
from DataManager import DataManager
from DataVisualizer import DataVisualizer
from Tokenizer import Tokenizer
from SequenceAnalyzer import SequenceAnalyzer

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
        
        attention_output = MultiHeadAttention(num_heads=8, key_dim=self.EMBEDDING_DIM, value_dim=self.EMBEDDING_DIM, name='self_attention_to_chat_text')(positional_output, positional_output)
        residual_output = Add(name='residual_connection_of_chat_text')([positional_output, attention_output])
        normalised_output = LayerNormalization(name='normalization_of_chat_text')(residual_output)
        _, hidden_state, cell_state = LSTM(self.HIDDEN_DIM, return_sequences=False, return_state=True, name='summarization_of_chat_text')(normalised_output)

        return chat_text, normalised_output, hidden_state, cell_state

    def define_decoder(self, processed_chat_text, summary_hidden_state, summary_cell_state):
        # handle previous sequence
        prev_seq = Input(shape=(self.MAX_SEQ_LENGTH,), name='prev_seq_input')
        embedding_output = Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM, mask_zero=True, name='embedding_of_prev_seq')(prev_seq)
        
        positional_layer = PositionalEncoding(self.MAX_SEQ_LENGTH, self.EMBEDDING_DIM, name='positional_encoding_of_prev_seq')
        positional_output = positional_layer(embedding_output)

        attention_output = Attention(name='self_attention_to_prev_seq')([positional_output, positional_output])
        residual_of_prev_seq = Add(name='residual_connection_of_prev_seq')([positional_output, attention_output])

        # handle emotion
        emotion = Input(shape=(1,), name='emotion_input')
        emotion_embedding = Embedding(len(list(EmotionStates)), self.EMBEDDING_DIM, name='embedding_of_emotion')(emotion)

        # multiple lstm
        _, lstm_chat_text_hidden_state, lstm_chat_text_cell_state = LSTM(self.HIDDEN_DIM, return_sequences=False, return_state=True, name='lstm_of_chat_text')(processed_chat_text, initial_state=[summary_hidden_state, summary_cell_state])
        _, lstm_emotion_hidden_state, lstm_emotion_cell_state = LSTM(self.HIDDEN_DIM, return_sequences=False, return_state=True, name='lstm_of_emotion')(emotion_embedding, initial_state=[lstm_chat_text_hidden_state, lstm_chat_text_cell_state])
        lstm_seq, _, lstm_state = LSTM(self.HIDDEN_DIM, return_sequences=True, return_state=True, name='lstm_of_prev_seq')(residual_of_prev_seq, initial_state=[lstm_emotion_hidden_state, lstm_emotion_cell_state])

        dense_of_seq = Dense(self.VOCAB_SIZE, name='dense_of_seq', activation='softmax')(lstm_seq)
        dense_of_state = Dense(self.VOCAB_SIZE, name='dense_of_state', activation='softmax')(lstm_state)

        return emotion, prev_seq, dense_of_seq, dense_of_state

    def define_model(self):
        if self.MODEL is None:
            chat_text, processed_chat_text, summary_hidden_state, summary_cell_state = self.define_encoder()
            emotion, prev_seq, dense_of_seq, dense_of_state = self.define_decoder(processed_chat_text, summary_hidden_state, summary_cell_state) 
            self.MODEL = Model([chat_text, emotion, prev_seq], [dense_of_seq, dense_of_state])
            # self.MODEL.name = self.MODEL_NAME
            
            logging("info","Model defined.")

            self.save_model()

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

        chat_text, emotion, prev_seq, _, _ = DataManager.preprocess_data(chat_text, text_response, emotion, self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)   

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
    
    def test_model(self, dataset_filename):
        chat_text, text_response, emotion = DataManager.prepare_data(dataset_filename)
        
        # Preprocess data
        chat_text_input, emotion_input, prev_seq, output_seq, output_state = DataManager.preprocess_data(chat_text, text_response, emotion, self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)

        loss, dense_of_seq_loss, dense_of_state_loss, dense_of_seq_accuracy, dense_of_state_accuracy = self.MODEL.evaluate([chat_text_input, emotion_input, prev_seq], [output_seq, output_state])
        logging("info", f"Model Evaluation\n\tloss: {loss}\n\tdense_of_seq_loss: {dense_of_seq_loss}\n\tdense_of_state_loss: {dense_of_state_loss}\n\tdense_of_seq_accuracy: {dense_of_seq_accuracy}\n\tdense_of_state_accuracy: {dense_of_state_accuracy}")

    def train_model(self, dataset_filename):
        chat_text, text_response, emotion = DataManager.prepare_data(dataset_filename)
        
        # Preprocess data
        chat_text_input, emotion_input, prev_seq, output_seq, output_state = DataManager.preprocess_data(chat_text, text_response, emotion, self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)

        # Train the model
        self.MODEL.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Define validation split for evaluation during training
        validation_split = 0.2  # 20% of training data for validation

        # Train the model with validation split
        history = self.MODEL.fit([chat_text_input, emotion_input, prev_seq], [output_seq, output_state], batch_size=64, epochs=10, validation_split=validation_split)

        DataVisualizer.plot_train_history(history.history, 'dense_of_seq', 'Model Training')
        DataVisualizer.plot_train_history(history.history, 'val_dense_of_seq', 'Model Validation')
        DataVisualizer.plot_train_history(history.history, 'dense_of_state', 'Model Training')
        DataVisualizer.plot_train_history(history.history, 'val_dense_of_state', 'Model Validation')

        logging("info", "Model trained.")    

        self.save_model()

    #   GENERTE RESPONSE

    def sequence_to_text(self, prev_seq):
        # Convert the sequence of token indices to text
        response_tokens = [self.TOKENIZER.TOKENIZER.index_word.get(idx.numpy(), "") for idx in tf.reshape(prev_seq[:,:,1:], (-1,))]
        response_tokens = [token for token in response_tokens if token!=""]

        # Remove tokens after <end> token
        end_index = response_tokens.index('<end>') if '<end>' in response_tokens else len(response_tokens)
        response_text = ' '.join(response_tokens[:end_index])
        return response_text

    def generate_response_with_greedy_approach(self, chat_text_str, emotion_str):
        chat_text_input, emotion_input, prev_seq, _, _ = DataManager.preprocess_data([chat_text_str], [""], [emotion_str], self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)
        prev_seq = tf.tensor_scatter_nd_update(prev_seq, indices=[[0, 1]], updates=[0])

        # DataVisualizer.print_tensor_dict("Input to model", {"chat_text": chat_text_input, "emotion": emotion_input, "prev_seq": prev_seq})

        for i in range(1, self.MAX_SEQ_LENGTH):
            # Predict the next token
            _, predicted_state = self.MODEL.predict([chat_text_input, emotion_input, prev_seq], verbose=0)
            
            # Get the token index with the highest probability (greedy approach)
            predicted_token_index = np.argmax(predicted_state[0, :])
            # print("TOKEN = ", predicted_token_index)
            
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
        tokenizer_length = self.TOKENIZER.length()

        # Preprocess data 
        chat_text_input, emotion_input, prev_seq, _, _ = DataManager.preprocess_data([chat_text_str], [""], [emotion_str], self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)
        prev_seq = tf.tensor_scatter_nd_update(prev_seq, indices=[[0, 1]], updates=[0])

        # DataVisualizer.print_tensor_dict("Input to model", {"chat_text": chat_text_input, "emotion": emotion_input, "prev_seq": prev_seq})

        # Initialize beam search set
        beam_list = [(prev_seq, 1.0)]

        # Initialize the final generated sequences
        final_sequences_list = []

        # Main loop for generating sequences
        for i in range(self.MAX_SEQ_LENGTH-1):
            candidates_list = []
            completed_sequences_list = []
            for prev_seq, score in beam_list:

                # Predict the next token probabilities
                _, predicted_state = self.MODEL.predict([chat_text_input, emotion_input, prev_seq], verbose=0)

                # Get the top tokens with their probabilities
                top_token_indices = np.argsort(predicted_state[0])[-beam_width:]
                top_token_indices = top_token_indices[top_token_indices != 0]

                for token_index in top_token_indices:
                    # Create a candidate sequence
                    candidate_seq = np.copy(prev_seq)

                    # Find the position where the next token should be inserted
                    next_token_position = np.where(candidate_seq == 0)[1][0]

                    # Insert the token at the correct position
                    candidate_seq[0, next_token_position] = token_index

                    # # Calculate the score for the candidate sequence
                    # candidate_score = score - np.log(predicted_state[0, token_index])       # negative likelyhood

                    # # Add penalty if the current token is equal to the previous token
                    # if next_token_position > 0 and candidate_seq[0, next_token_position] == candidate_seq[0, next_token_position - 1]:
                    #     penalty = -15  # You can adjust the penalty factor as needed
                    #     candidate_score += penalty

                    # # Normalize score by dividing by the length of the sequence raised to a power
                    # length_factor = 0.7  # You can adjust this factor as needed
                    # candidate_score /= len(candidate_seq[0])**length_factor
                    candidate_score = score * SequenceAnalyzer.calculate_score(candidate_seq[0], chat_text_input[0].numpy(), tokenizer_length, predicted_state[0, token_index])

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
