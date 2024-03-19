from DialogueGenerator import DialogueGenerator

# Instantiate the Chatbot class
dialogue_generator = DialogueGenerator()

# Define model
# dialogue_generator.define_model()
# print(dialogue_generator.model)

# Train the model
# dialogue_generator.model_visualization()
dialogue_generator.train_and_test('Datasets/Topical_Chat_1.csv', 5)

# Inspection
# dialogue_generator.inspect_layer_outputs()

