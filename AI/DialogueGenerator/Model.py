from keras.models import load_model

# Load the Keras model
model = load_model('dialogue_generator_model.keras')

print(model.summary())

