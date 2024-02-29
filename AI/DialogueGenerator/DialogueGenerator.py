import pandas as pd
import numpy as np
import tensorflow as tf
from enum import Enum
from States import EmotionStates, get_emotion_index
from tensorflow.keras.layers import Input, Embedding, Concatenate, Add, Attention, MultiHeadAttention, LSTM, Dense, LayerNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend

class DialogueGenerator:

    def __init__(self):
        self.MAX_SEQ_LENGTH = 50  # Maximum sequence length for input and output
        self.VOCAB_SIZE = 10000   # Vocabulary size
        self.EMBEDDING_DIM = 300  # Embedding dimension
        self.HIDDEN_DIM = 512     # Hidden dimension for LSTM layers
        self.EMOTION_SIZE = 1

    def display_top_rows(self, df):
        print("Top 5 rows of the dataset:\n%s", df.head())
            
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
        encoder_inputs_1 = tf.keras.preprocessing.sequence.pad_sequences(chat_text_sequences, maxlen=self.MAX_SEQ_LENGTH, padding='post')
        decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(text_response_sequences, maxlen=self.MAX_SEQ_LENGTH, padding='post')
        print("\nPadding...")
        print("Encoder input[0] = ", encoder_inputs_1[0])
        print("Decoder input[0] = ", decoder_inputs[0])

        # Shift targets for teacher forcing
        decoder_outputs = np.zeros_like(decoder_inputs)
        decoder_outputs[:, :-1] = decoder_inputs[:, 1:]                                    # ??? why shifting
        decoder_outputs[:, -1] = 0
        print("\nDecoder output[0] = ", decoder_outputs[0])

        # Map emotion strings to their corresponding enum values
        encoder_inputs_2 = [get_emotion_index(emo) for emo in emotion] 
        print("\nemotion_index[0] = ", encoder_inputs_2[0])
        
        print("Data preprocessed.")
        
        return encoder_inputs_1, encoder_inputs_2, decoder_inputs, decoder_outputs

    def define_encoder(self):
        print("\n\n-> ENCODER")

        print("\n- LAYER 0 - INPUT")
        encoder_chat_text_inputs = Input(shape=(self.MAX_SEQ_LENGTH,), name='encoder_input_1_chat_text')
        # encoder_emotion_inputs = Input(shape=(self.EMOTION_SIZE,), name='encoder_input_2_emotion')
        print("Encoder Chat Text Input Shape: ", encoder_chat_text_inputs.shape)
        # print("Encoder Emotion Input Shape: ", encoder_emotion_inputs.shape)
        
        print("\n- LAYER 1 - EMBEDDING")
        encoder_embedding = Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM, mask_zero=True, name='dense_embedding_of_chat_text')(encoder_chat_text_inputs)
        print("Encoder Embedding Output Shape: ", encoder_embedding.shape)

        print("\n- LAYER 2 - (ORDER) - POSITIONAL EMBEDDING")

        # Generate positional encodings
        # position_encoding = tf.range(start=0, limit=self.MAX_SEQ_LENGTH, delta=1)
        position_encoding = tf.expand_dims(tf.cast(tf.range(self.MAX_SEQ_LENGTH), dtype=tf.float32), axis=-1) / tf.cast((self.EMBEDDING_DIM - 1), dtype=tf.float32)
        position_encoding = position_encoding * tf.cast(tf.ones((1, self.EMBEDDING_DIM)), dtype=tf.float32)
        print("Positional encoding = ", position_encoding)

        # Add positional embedding to the dense vector
        positional_output = Add(name='positional_embedding_of_chat_text')([encoder_embedding, tf.expand_dims(position_encoding, 0)])

        print("Positional Embeddings Output Shape: ", positional_output.shape)

        print("\n- LAYER 3 - (CONTEXT) - MULTI HEAD ATTENTION")
        attention = MultiHeadAttention(num_heads=8, key_dim=self.EMBEDDING_DIM, value_dim=self.EMBEDDING_DIM, name='multi_head_attention_to_chat_text')(positional_output, key=positional_output, value=positional_output)
        print("Attention Output Shape: ", attention.shape)

        print("\n- LAYER 4 - ADDING RESIDUAL CONNECTION")
        residual_output = Add(name='residual_positional_output_plus_attention_output')([positional_output, attention])
        print("Residual Addition Shape: ", residual_output.shape)

        print("\n- LAYER 5 - NORMALIZATION")
        normalised_output = LayerNormalization(name='normalization_of_residual_output')(residual_output)
        print("Normalization Shape: ", normalised_output.shape)

        print("\n- LAYER 6 - (HISTORY) - LSTM")
        encoder_outputs, state_h, state_c = LSTM(self.HIDDEN_DIM, return_sequences=True, return_state=True, name='lstm_of_chat_text')(normalised_output)
        encoder_states = [state_h, state_c]
        print("LSTM Output Shape: ", encoder_outputs.shape)

        print("\nEncoder defined.")

        return encoder_chat_text_inputs, encoder_states, encoder_outputs

    def define_decoder(self, encoder_states, encoder_outputs):
        print("\n\n-> DECODER")

        # LAYER 1 - INPUT
        print("\n- LAYER 1 - INPUT")
        decoder_inputs = Input(shape=(self.MAX_SEQ_LENGTH,), name='decoder_input')

        # LAYER 2 - EMBEDDING
        print("\n- LAYER 2 - EMBEDDING")
        decoder_embedding = Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM, mask_zero=True, name='embedding_of_decoder_input')(decoder_inputs)
        print("Decoder Embedding Output Shape: ", decoder_embedding.shape)
        
        # LAYER 3 - LSTM LAYER
        print("\n- LAYER 3 - LSTM")
        lstm_context = LSTM(self.HIDDEN_DIM, return_sequences=True, return_state=True, name='lstm1_of_decoder_embedding_and_encoder_states')
        
        # Initialize context memory with encoder final states
        _, context_state_h, context_state_c = lstm_context(decoder_embedding, initial_state=encoder_states)
        context_state = [context_state_h, context_state_c]
        
        # LAYER 4 - LSTM
        print("\n- LAYER 4 - LSTM")
        lstm_decoder = LSTM(self.HIDDEN_DIM, return_sequences=True, return_state=True, name='lstm2_of_decoder_embedding_and_prev')
        decoder_outputs, _, _ = lstm_decoder(decoder_embedding, initial_state=context_state)
        print("LSTM Decoder Output Shape: ", decoder_outputs.shape)
        
        # LAYER 5 - ATTENTION
        print("\n- LAYER 5 - ATTENTION")
        attention = Attention(name='attention_to_prev_and_encoder_outputs')
        attention_output = attention([decoder_outputs, encoder_outputs])
        print("Attention Output Shape: ", attention_output.shape)
        
        # LAYER 6 - CONCATENATION 
        print("\n- LAYER 6 - CONCATENATION")
        decoder_concat = Concatenate(axis=-1, name='cancat_decoder_outputs_and_prev')([decoder_outputs, attention_output])
        print("Concatenation Output Shape: ", decoder_concat.shape)
        
        # LAYER 7 - DENSE 
        print("\n- LAYER 7 - DENSE")
        decoder_dense = Dense(self.VOCAB_SIZE, name='dense_of_prev')
        decoder_outputs = decoder_dense(decoder_concat)
        print("Dense Output Shape: ", decoder_outputs.shape)
        
        print("\nDecoder defined.\n")
        
        return decoder_inputs, decoder_outputs

    def define_model(self):
        concatenated_inputs, encoder_states, encoder_outputs = self.define_encoder()
        decoder_inputs, decoder_outputs = self.define_decoder(encoder_states, encoder_outputs)
        self.model = Model([concatenated_inputs, decoder_inputs], decoder_outputs)
        
        print("\nModel defined.")
        print("Model Summary:")
        self.model.summary()

    def print_layer_output(self, epoch, logs):
        # Input tensor (you need to provide your input data here)
        chat_text, emotion, _, _ = self.preprocess_data([self.zeroth_row['chat_text']], [self.zeroth_row['text_response']], [self.zeroth_row['emotion']])
        zeroth_row_input_tensor = [chat_text, emotion]

        # Get output tensor of each layer
        for layer_index, layer in enumerate(self.model.layers):
            layer_output_function = backend.function([self.model.input], [layer.output])
            output_tensor = layer_output_function([zeroth_row_input_tensor])[0]
            print(f'Layer {layer_index} output:')
            print(output_tensor)
            print('\n')

    def train_model(self, encoder_inputs, decoder_inputs, decoder_outputs, batch_size, epochs):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        print_callback = LambdaCallback(on_epoch_end=self.print_layer_output)
        self.model.fit([encoder_inputs, decoder_inputs], decoder_outputs, batch_size=batch_size, 
            epochs=epochs, validation_split=0.2, callbacks=[print_callback])
        
        print("\nModel trained.\n")
        
    def create_tokenizer(self, chat_text, text_response):
        # Concatenate chat_text and text_response
        all_texts = chat_text + text_response
        print("Texts concatenated.\n")
        
        # Create tokenizer and fit on all texts
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.VOCAB_SIZE - 3, oov_token='<OOV>')
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

    def extract_data_from_dataframe(self, df):
        chat_text = df['chat_text'].tolist()
        text_response = df['text_response'].tolist()
        emotion = df['emotion'].tolist()
        print("\nData extracted from DataFrame.")
        return chat_text, text_response, emotion

    def create_train_and_save_model(self):
        # Load the dataset
        df = pd.read_csv('Conversation3.csv')
        self.display_top_rows(df)
        self.zeroth_row = df.iloc[0]

        # Extract data from the DataFrame
        chat_text, text_response, emotion = self.extract_data_from_dataframe(df)

        self.create_tokenizer(chat_text, text_response)
        
        # Preprocess data
        encoder_inputs_1, encoder_inputs_2, decoder_inputs, decoder_outputs = self.preprocess_data(chat_text, text_response, emotion)

        # Define and compile the model
        self.define_model()

        # Train the model
        self.train_model(encoder_inputs_1, decoder_inputs, decoder_outputs, batch_size=64, epochs=1)           

        # Save the tokenizer
        self.save_tokenizer()

        # Save the trained model
        self.model.save("conversation_model.keras")
        print("Trained model saved.\n")

    def generate_response_with_greedy_approach(self, chat_text, emotion_str, max_seq_length):
        print("\n\n-> GENERATE RESPONSE")

        # Preprocess input text
        chat_text_sequence = tokenizer.texts_to_sequences([chat_text])
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
        target_seq[0, 0] = tokenizer.word_index['<start>']
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
                token = tokenizer.index_word.get(index, None)
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