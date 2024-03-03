from transformers import GPT2LMHeadModel, GPT2Tokenizer

class DialogueGenerator:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def generate_response(self, previous_chat, emotion, action):
        input_text = "Previous chat: {} Emotion: {} Action: {}".format(previous_chat, emotion, action)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Generate response
        output = self.model.generate(input_ids, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.9, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response

# Example usage:
dialogue_generator = DialogueGenerator()
previous_chat = "Hello, how are you?"
emotion = "happy"
action = "smiling"

response = dialogue_generator.generate_response(previous_chat, emotion, action)
print("Generated Response:", response)
