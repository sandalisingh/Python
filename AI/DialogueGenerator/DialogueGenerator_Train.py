from DialogueGenerator import DialogueGenerator

# Instantiate the Chatbot class
dialogue_generator = DialogueGenerator()

# Define model
# dialogue_generator.define_model()
# print(dialogue_generator.model)

# Train the model
# dialogue_generator.model_visualization()
dialogue_generator.train_model('Datasets/Conversation_org.csv')
dialogue_generator.test_model('Datasets/Conversation_org.csv')

# Inspection
# dialogue_generator.inspect_layer_outputs()

