from DialogueGenerator import DialogueGenerator
import tensorflow as tf

# Instantiate the Chatbot class
dialogue_generator = DialogueGenerator()

# Load the model
# model_path = 'dialogue_generator_model'  # Path to the directory where the model is saved
# loaded_model = tf.saved_model.load(model_path)

# Train the model
dialogue_generator.create_train_and_save_model()

# Inspection
# dialogue_generator.inspect_layer_outputs()

