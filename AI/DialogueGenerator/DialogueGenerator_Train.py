from DialogueGenerator import DialogueGenerator

# Instantiate the Chatbot class
dialogue_generator = DialogueGenerator()

# Train the model
# dialogue_generator.model_visualization()
dialogue_generator.train_and_test('Datasets/Conversation_org.csv', 5)

# Inspection
# dialogue_generator.inspect_layer_outputs()

