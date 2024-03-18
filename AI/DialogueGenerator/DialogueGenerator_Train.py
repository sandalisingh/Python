from DialogueGenerator import DialogueGenerator

# Instantiate the Chatbot class
dialogue_generator = DialogueGenerator()

# Define model
# dialogue_generator.define_model()
# print(dialogue_generator.model)

# Train the model
# dialogue_generator.model_visualization()
dialogue_generator.train_and_test('Datasets/Dataset_69K_Kaggle_7.csv', 5)

# Inspection
# dialogue_generator.inspect_layer_outputs()

