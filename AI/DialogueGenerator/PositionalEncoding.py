import numpy as np
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, embedding_dim, name='positional_encoding', **kwargs):
        super(PositionalEncoding, self).__init__(name=name, **kwargs)
        self.MAX_SEQ_LENGTH = max_seq_len
        self.EMBEDDING_DIM = embedding_dim
        self.position_encoding = self.calculate_positional_encoding()

    def calculate_positional_encoding(self):
        position_encoding = np.zeros((self.MAX_SEQ_LENGTH, self.EMBEDDING_DIM), dtype=np.float32)
        for pos in range(self.MAX_SEQ_LENGTH):
            for i in range(self.EMBEDDING_DIM):
                position_encoding[pos, i] = np.sin(pos / np.power(10000, (2 * i) / self.EMBEDDING_DIM))
                position_encoding[pos, i] = np.cos(pos / np.power(10000, (2 * i + 1) / self.EMBEDDING_DIM))
        
        # print("Positional Encoding:")
        # print(position_encoding)
        return tf.convert_to_tensor(position_encoding)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        tiled_position_encoding = tf.tile(tf.expand_dims(self.position_encoding, 0), [batch_size, 1, 1])
        
        # print("Tiled Positional Encoding:")
        # print(tiled_position_encoding)
        
        result = inputs + tiled_position_encoding
        
        # print("Result after adding Positional Encoding:")
        # print(result)
        
        return result

# Register the custom layer
tf.keras.utils.get_custom_objects()['PositionalEncoding'] = PositionalEncoding