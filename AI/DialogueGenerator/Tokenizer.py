import tensorflow as tf
import pickle
from States import logging

class Tokenizer():

    def __init__(self, vocab_size=10000):
        self.TOKENIZER = None
        self.TOKENIZER_NAME = "tokenizer.pkl"
        self.VOCAB_SIZE = vocab_size
        self.START_TOKEN = 9998
        self.END_TOKEN = 9999

        self.init_tokenizer()

    def init_tokenizer(self):
        try:
            with open(self.TOKENIZER_NAME, 'rb') as tokenizer_file:
                self.TOKENIZER = pickle.load(tokenizer_file)
            logging("info", "Tokenizer loaded.")
        except Exception as e:
            logging("error", str(e))
            self.TOKENIZER = None

    def fit_tokenizer(self, chat_text, text_response):
        # Concatenate chat_text and text_response
        all_texts = chat_text + text_response
        print("Texts concatenated.\n")

        if self.TOKENIZER is None:
            # Create tokenizer and fit on all texts
            self.TOKENIZER = tf.keras.preprocessing.text.Tokenizer(num_words=self.VOCAB_SIZE - 3, oov_token='<OOV>', filters='')
            self.TOKENIZER.fit_on_texts(all_texts)
            self.TOKENIZER.word_index['<start>'] = self.TOKENIZER.num_words + 1
            self.TOKENIZER.word_index['<end>'] = self.TOKENIZER.num_words + 2
            
            # Manually add <start> and <end> to index_word
            self.TOKENIZER.index_word[self.TOKENIZER.word_index['<start>']] = '<start>'
            self.TOKENIZER.index_word[self.TOKENIZER.word_index['<end>']] = '<end>'
            
            logging("info", "Tokenizer created.")
        else:
            self.TOKENIZER.fit_on_texts(all_texts)
            logging("info", "Tokenizer fitted on new data.")
            
        self.save_tokenizer()
        print("Tokenizer size = ", self.VOCAB_SIZE)
        # self.print_tokenizer()

    def save_tokenizer(self):
        with open(self.TOKENIZER_NAME, 'wb') as tokenizer_file:
            pickle.dump(self.TOKENIZER, tokenizer_file)
        logging("info", "Tokenizer saved at "+self.TOKENIZER_NAME)
        return tokenizer_file
    
    def print_tokenizer(self):
        if self.TOKENIZER is not None:
            print("Tokens = {")
            for i, (word, index) in enumerate(self.TOKENIZER.word_index.items()):
                # Formatting each token and index pair
                token_format = f'({index}, \'{word}\')'
                print(token_format + ", ", end="")

            print("}")

    def length(self):
        if self.TOKENIZER is not None and self.TOKENIZER.word_index:
            return len(self.TOKENIZER.word_index) + 1 
        else:
            return 0
    
