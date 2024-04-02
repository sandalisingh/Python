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

    def set_start_and_end_tokens(self):
        self.TOKENIZER.word_index['<start>'] = self.START_TOKEN
        self.TOKENIZER.word_index['<end>'] = self.END_TOKEN
        
        self.TOKENIZER.index_word[self.START_TOKEN] = '<start>'
        self.TOKENIZER.index_word[self.END_TOKEN] = '<end>'

    def filter_tokens_by_vocab_size(self):
        # Filter tokens in word index
        self.TOKENIZER.word_index = {word: index for word, index in self.TOKENIZER.word_index.items() if index < self.VOCAB_SIZE-2}
        
        # Filter tokens in index word
        self.TOKENIZER.index_word = {index: word for index, word in self.TOKENIZER.index_word.items() if index < self.VOCAB_SIZE-2}

    def fit_tokenizer(self, chat_text, text_response):
        # Concatenate chat_text and text_response
        all_texts = chat_text + text_response
        # print("Texts concatenated.\n")

        if self.TOKENIZER is None:
            # Create tokenizer and fit on all texts
            self.TOKENIZER = tf.keras.preprocessing.text.Tokenizer(num_words=self.VOCAB_SIZE - 3, oov_token='<OOV>', filters='')
            logging("info", "New tokenizer created.")  

        if len(self.TOKENIZER.word_index) < self.VOCAB_SIZE - 2:
            self.TOKENIZER.fit_on_texts(all_texts)
            self.filter_tokens_by_vocab_size()
            self.set_start_and_end_tokens()
            logging("info", "Tokenizer fitted on new data.")
        else:
            logging("info", "Tokenizer is already full. Not fitting on new data.")
            
        self.save_tokenizer()
        # print("Tokenizer size = ", self.VOCAB_SIZE)
        # self.print_tokenizer()

    def save_tokenizer(self):
        with open(self.TOKENIZER_NAME, 'wb') as tokenizer_file:
            pickle.dump(self.TOKENIZER, tokenizer_file)
        logging("info", "Tokenizer saved at "+self.TOKENIZER_NAME)
        return tokenizer_file
    
    def print_tokenizer(self, length=10000):
        if self.TOKENIZER is not None:
            print("Tokens = {")
            for i, (word, index) in enumerate(self.TOKENIZER.word_index.items()):
                if i>=length:
                    break

                # Formatting each token and index pair
                token_format = f'({index}, \'{word}\')'
                print(token_format + ", ", end="")

            print("}")

    def length(self):
        if self.TOKENIZER is not None and self.TOKENIZER.word_index:
            return len(self.TOKENIZER.word_index) + 1 
        else:
            return 0
    
