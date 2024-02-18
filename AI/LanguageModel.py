import numpy as np
import tensorflow as tf

# Load and preprocess data
text = open('corpus.txt', 'r').read()  # Load your text data here
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
text_length = len(text)
vocab_size = len(chars)

# Hyperparameters
sequence_length = 100  # Length of input sequences
batch_size = 64
embedding_dim = 256
rnn_units = 1024  # Number of RNN units
num_epochs = 50

# Convert text to sequences of indices
sequences = []
for i in range(0, text_length - sequence_length, sequence_length):
    sequences.append([char_to_idx[ch] for ch in text[i:i + sequence_length]])

# Create input-output pairs
input_seqs = []
output_seqs = []
for seq in sequences:
    input_seqs.append(seq[:-1])
    output_seqs.append(seq[1:])

# Convert to numpy arrays
input_seqs = np.array(input_seqs)
output_seqs = np.array(output_seqs)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Train the model
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    for batch_idx in range(0, input_seqs.shape[0] - batch_size, batch_size):
        input_batch = input_seqs[batch_idx:batch_idx + batch_size]
        output_batch = output_seqs[batch_idx:batch_idx + batch_size]
        model.train_on_batch(input_batch, output_batch)
    model.reset_states()

# Save the model
model.save('language_model.h5')
