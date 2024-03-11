import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from Dataset_Preprocess import preprocessing_dataset, balance_dataset, EmotionList

# List of dataset filenames
dataset_files = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]

class EmotionClassifier:
    def __init__(self, dataset_files):
        self.dataset_files = dataset_files

    def load_and_preprocess_datasets(self):
        # Load all datasets, preprocess, and concatenate them
        data_list = []
        for dataset_file in self.dataset_files:
            print(f"\nLoading and preprocessing {dataset_file} ...")
            data = pd.read_csv(dataset_file)
            data = preprocessing_dataset(data)
            data = balance_dataset(data, 'emotion')
            data_list.append(data)
            data.to_csv(dataset_file, index=False)
            print(f"{dataset_file} loaded, preprocessed, balanced and saved\n")

        # Concatenate datasets
        self.data = pd.concat(data_list, ignore_index=True)
        print("Three datasets concatenated")

        # Print counts for each emotion
        print("\nCounts for each emotion:")
        print(self.data['emotion'].value_counts())

    def split_data(self):
        # Split data into features (text) and target (emotions)
        X = self.data['text']
        Y = self.data['emotion']
        print("Splitted data into features(text) and target(emotion)")

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print("Data split into training and testing completed")

    def define_model_pipeline(self):
        # Define the model pipeline
        self.model = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(max_iter=1000, multi_class='auto')
        )
        print("\nModel pipeline defined - TfidfVectorizer, LogisticRegression")

    def train_model(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

    def evaluate_model(self):
        # Evaluate the model
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print("\nModel evaluation completed.")

    def save_model(self, filename='emotion_classifier_model.joblib'):
        # Save the trained model
        joblib.dump(self.model, 'emotion_classifier_model.joblib')
        print("\nTrained model saved.")
