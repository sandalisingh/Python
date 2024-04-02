class SequenceAnalyzer:
    @staticmethod
    def calculate_diversity(tokens, seq_lt):
        unique_tokens = set(tokens)
        num_unique_tokens = len(unique_tokens)-1
        
        normalized_unique_tokens = num_unique_tokens / seq_lt
        
        return normalized_unique_tokens

    @staticmethod
    def calculate_responsiveness(generated_response, input_tokens, seq_lt):
        # Count the number of input tokens present in the generated response
        no_of_matches = sum(token in generated_response for token in input_tokens)
        
        # Normalize the count to a value between 0 and 1 based on the total number of input tokens
        responsiveness_score = no_of_matches / seq_lt 

        return responsiveness_score

    @staticmethod
    def calculate_metrics(candidate, input_text, seq_lt):
        # Calculate diversity
        num_unique_tokens = SequenceAnalyzer.calculate_diversity(candidate, seq_lt)
        
        # Calculate responsiveness
        responsiveness = SequenceAnalyzer.calculate_responsiveness(candidate, input_text, seq_lt)
        
        return num_unique_tokens, responsiveness

    @staticmethod
    def calculate_score(candidate, input_text, seq_length, probability, weight_diversity=0.5, weight_responsiveness=0.5, weight_probability=0.5):
        input_text = set(input_text)
        candidate = set(candidate)

        num_unique_tokens, responsiveness = SequenceAnalyzer.calculate_metrics(candidate, input_text, seq_length)
        
        # Calculate combined score using weighted sum
        combined_score = (weight_diversity * num_unique_tokens) + (weight_responsiveness * responsiveness) + (weight_probability * probability)
        
        # Normalize the combined score
        max_score = weight_diversity + weight_responsiveness + weight_probability
        normalized_score = combined_score / max_score
        
        return combined_score