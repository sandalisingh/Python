class SequenceAnalyzer:
    @staticmethod
    def calculate_diversity(tokens, tokenizer_length):
        unique_tokens = set(tokens)
        num_unique_tokens = len(unique_tokens)
        
        normalized_unique_tokens = num_unique_tokens / tokenizer_length
        
        return normalized_unique_tokens

    @staticmethod
    def calculate_responsiveness(generated_response, input_tokens):
        # Count the number of input tokens present in the generated response
        no_of_matches = sum(token in generated_response for token in input_tokens)
        
        # Normalize the count to a value between 0 and 1 based on the total number of input tokens
        responsiveness_score = no_of_matches / len(input_tokens) if len(input_tokens) > 0 else 0

        return responsiveness_score

    @staticmethod
    def calculate_metrics(candidate, input_text, tokenizer_length):
        # Calculate diversity
        num_unique_tokens = SequenceAnalyzer.calculate_diversity(candidate, tokenizer_length)
        
        # Calculate responsiveness
        responsiveness = SequenceAnalyzer.calculate_responsiveness(candidate, input_text)
        
        return num_unique_tokens, responsiveness

    def calculate_score(candidate, input_text, tokenizer_length, weight_diversity=0.5, weight_responsiveness=0.5):
        input_text = input_text[input_text != 0]

        num_unique_tokens, responsiveness = SequenceAnalyzer.calculate_metrics(candidate, input_text, tokenizer_length)
        
        # Calculate combined score using weighted sum
        score = (weight_diversity * num_unique_tokens) + (weight_responsiveness * responsiveness)
        
        return score