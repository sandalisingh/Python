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
        DataVisualizer.get_arch_flowchat(self.MODEL, "model_graph_plot.png")

    def inspect_layer_outputs(self):
        chat_text, text_response, emotion = DataManager.prepare_data("Conversation3.csv")
        
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

    def evaluate_model(self, encoder_inputs, emotion_inputs, decoder_inputs, decoder_outputs):
        loss, accuracy = self.MODEL.evaluate([encoder_inputs, emotion_inputs, decoder_inputs], decoder_outputs)
        logging("info", f"Model Evaluation - Loss: {loss}, Accuracy: {accuracy}")
        return loss, accuracy

    def train_model(self, encoder_inputs, emotion_inputs, decoder_inputs, decoder_outputs, batch_size, epochs):
        self.MODEL.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Define validation split for evaluation during training
        validation_split = 0.2  # 20% of training data for validation

        # Train the model with validation split
        history = self.MODEL.fit([encoder_inputs, emotion_inputs, decoder_inputs], [decoder_outputs, None], batch_size=batch_size, 
                    epochs=epochs
                    # , validation_split=validation_split
                    )

        logging("info", "Model trained.")
    
    def create_train_and_save_model(self):
        chat_text, text_response, emotion = DataManager.prepare_data("TestData.csv")
        
        # Preprocess data
        encoder_inputs_chat_text, emotion_inputs, decoder_inputs, decoder_outputs = DataManager.preprocess_data(chat_text, text_response, emotion, self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)

        # Train the model
        self.train_model(encoder_inputs_chat_text, emotion_inputs, decoder_inputs, decoder_outputs, batch_size=64, epochs=1)       

        # Evaluate the model
        loss, accuracy = self.evaluate_model(encoder_inputs_chat_text, emotion_inputs, decoder_inputs, decoder_outputs)    

        # Plot and save the training history
        DataVisualizer.plot_loss_and_accuracy(loss, accuracy)

        self.save_model()

    #   GENERTE RESPONSE

    def sequence_to_text(self, prev_seq):
        # Convert the sequence of token indices to text
        response_tokens = [self.TOKENIZER.TOKENIZER.index_word.get(idx.numpy(), "<OOV>") for idx in tf.reshape(prev_seq, (-1,))]
        
        # Remove tokens after <end> token
        end_index = response_tokens.index('<end>') if '<end>' in response_tokens else len(response_tokens)
        response_text = ' '.join(response_tokens[:end_index])
        return response_text

    def generate_response_with_greedy_approach(self, chat_text_str, emotion_str):
        print("\n\n-> GENERATE RESPONSE")

        chat_text_input, emotion_input, _, prev_seq = DataManager.preprocess_data([chat_text_str], [""], [emotion_str], self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)

        for i in range(1, self.MAX_SEQ_LENGTH):
            # Predict the next token
            predictions = self.MODEL.predict([chat_text_input, emotion_input, prev_seq])
            
            # Get the token index with the highest probability (greedy approach)
            predicted_token_index = np.argmax(predictions[0, i - 1, :])
            
            # Update the previous sequence tensor
            updated_value = tf.constant(predicted_token_index, dtype=tf.int32)
            prev_seq = tf.tensor_scatter_nd_update(prev_seq, [[0, i]], [updated_value])

            # If the predicted token is an end token, break the loop
            if predicted_token_index == self.TOKENIZER.END_TOKEN:
                break
            
        response_text = self.sequence_to_text(prev_seq)

        logging("info", "Response generated")
        
        return response_text 

    def generate_response_with_beam_search(self, chat_text_str, emotion_str, beam_width=5):
        # Preprocess data
        chat_text_input, emotion_input, _, prev_seq = DataManager.preprocess_data([chat_text_str], [""], [emotion_str], self.VOCAB_SIZE, self.MAX_SEQ_LENGTH)
        
        # Update the first element from <end> to <start>
        # prev_seq = tf.tensor_scatter_nd_update(prev_seq, indices=[[0, 0]], updates=[self.TOKENIZER.START_TOKEN])
        
        print("prev_seq : ") 
        print(prev_seq) 

        # Initialize beam search set
        beam = {(tuple(prev_seq.numpy().flatten()), 0)}

        # Initialize the final generated sequences
        final_sequences = set()

        # Main loop for generating sequences
        for _ in range(self.MAX_SEQ_LENGTH):
            candidates = set()
            completed_sequences = set()
            for prev_seq, score in beam:
                # Predict the next token probabilities
                predictions = self.MODEL.predict([chat_text_input, emotion_input, prev_seq])

                # Get the top tokens with their probabilities
                # top_tokens = np.argsort(predictions[0, -1])[-beam_width:]
                top_tokens = np.argsort(predictions[0, -1])[-beam_width:][::-1]

                print("Top tokens:")
                print(top_tokens)

                for token in top_tokens:
                    print("Token:")
                    print(token)

                    # Create a candidate sequence
                    candidate_seq = np.copy(prev_seq)

                    # Find the position where the next token should be inserted
                    next_token_position = np.where(candidate_seq == 0)[0][0]

                    # Insert the token at the correct position
                    candidate_seq[0, next_token_position] = token

                    # Calculate the score for the candidate sequence
                    candidate_score = score - np.log(predictions[0, -1, token])

                    print("Candidate seq:")
                    print(candidate_seq)
                    print("Candidate score:")
                    print(candidate_score)

                    # Check if the sequence is complete
                    if token == self.TOKENIZER.END_TOKEN:
                        completed_sequences.append((tuple(candidate_seq.numpy().flatten()), candidate_score))
                    else:
                        candidates.append((tuple(candidate_seq.numpy().flatten()), candidate_score))

            # Select top candidates to continue beam search
            beam = set(sorted(candidates, key=lambda x: x[1])[:beam_width])

            # Check for completion of sequences
            final_sequences.update(completed_sequences)

            print("Beam search candidates:")
            print(beam)
            print("Completed sequences:")
            print(completed_sequences)
            print("Final sequences:")
            print(final_sequences)

            if final_sequences:
                final_sequences = set(sorted(final_sequences, key=lambda x: x[1]))
                best_seq = final_sequences.pop()[0]

                # Extract and print all final sequences
                all_sequences = [self.sequence_to_text(np.array([seq[0]])) for seq in final_sequences]
                print("All Final Sequences:")
                for seq in all_sequences:
                    print(seq)
            else:
                # Choose the best sequence from the last beam
                best_seq = beam.pop()[0]

            # Decode the best sequence into text
            response_text = self.sequence_to_text(np.array([best_seq]))

            print("Best sequence:")
            print(response_text)

        return response_text
