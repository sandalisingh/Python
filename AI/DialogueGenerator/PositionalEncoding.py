import numpy as np
import tensorflow as tf
import math

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, embedding_dim, name='positional_encoding', **kwargs):
        super(PositionalEncoding, self).__init__(name=name, **kwargs)
        self.supports_masking = True
        self.MAX_SEQ_LENGTH = max_seq_len
        self.EMBEDDING_DIM = embedding_dim
        self.position_encoding = self.calculate_positional_encoding()

    def calculate_positional_encoding(self):
        position_encoding = np.zeros((self.MAX_SEQ_LENGTH, self.EMBEDDING_DIM))
        for pos in range(self.MAX_SEQ_LENGTH):
            for i in range(0, self.EMBEDDING_DIM, 2):
                position_encoding[pos, i] = np.sin(pos / (10000 ** (i / self.EMBEDDING_DIM)))
                position_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / self.EMBEDDING_DIM)))
        return tf.constant(position_encoding, dtype=tf.float32)

    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]

        # Handle cases where mask is None
        if mask is None:
            # Create a mask that zeros out padding values (0)
            mask = tf.where(tf.not_equal(inputs, 0), 1.0, 0.0)

        # Expand mask for broadcasting along the embedding dimension
        mask = tf.expand_dims(mask, axis=-1)
        
        # Mask out padding before tiling
        masked_inputs = inputs * tf.cast(mask, dtype=tf.float32)
        tiled_position_encoding = tf.tile(tf.expand_dims(self.position_encoding, 0), [batch_size, 1, 1])

        # Apply masking to positional encoding as well
        masked_position_encoding = tiled_position_encoding * tf.cast(mask, dtype=tf.float32)
        result = masked_inputs + masked_position_encoding

        return result

    def compute_mask(self, inputs, mask=None):  
        return mask

# Register the custom layer
tf.keras.utils.get_custom_objects()['PositionalEncoding'] = PositionalEncoding