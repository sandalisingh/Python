import pickle

# Load the tokenizer
tokenizer_path = "tokenizer.pkl"
with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
print("Tokenizer loaded from %s.", tokenizer_path)

# Access the word_index attribute of the tokenizer
word_index = tokenizer.word_index

# Print all tokens in the desired format
print("Tokens = {")
for i, (word, index) in enumerate(word_index.items()):
    # Formatting each token and index pair
    token_format = f'({index}, \'{word}\')'
    print(token_format + ", ", end="")

print("}")

print("<start> : ",tokenizer.word_index['<start>'])
print("<end> : ",tokenizer.word_index['<end>'])
print("9999 : ",tokenizer.index_word.get(9999, None))
print("9998 : ",tokenizer.index_word.get(9998, None))
print("123 : ",tokenizer.index_word.get(123, None))
print("12 : ",tokenizer.index_word.get(12, None))
print("1 : ",tokenizer.index_word.get(1, None))