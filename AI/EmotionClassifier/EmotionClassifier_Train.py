from EmotionClassifier import EmotionClassifier

# List of dataset filenames
dataset_files = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]

# Create EmotionClassifier instance
emotion_classifier = EmotionClassifier(dataset_files)

# Load, preprocess, and balance datasets
emotion_classifier.load_and_preprocess_datasets()

# Split data into training and testing sets
emotion_classifier.split_data()

# Define the model pipeline
emotion_classifier.define_model_pipeline()

# Train the model
emotion_classifier.train_model()

# Evaluate the model
emotion_classifier.evaluate_model()

# Save the trained model
emotion_classifier.save_model()