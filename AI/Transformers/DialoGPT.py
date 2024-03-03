from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def chat_with_dialogpt():
    # Load pretrained model and tokenizer
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Chatting loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Tokenize input and generate response
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        print("Bot:", response)

if __name__ == "__main__":
    chat_with_dialogpt()
