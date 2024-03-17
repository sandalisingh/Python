import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from Dataset_Preprocess import preprocessing_dataset, balance_dataset, EmotionList
from States import logging, EmotionStates

class EmotionClassifier:
    def __init__(self):
        self.MODEL_NAME = 'emotion_classifier_model.joblib'
        self.MODEL = self.load_model()

    def load_and_preprocess_datasets(self, dataset_files):
        # Load all datasets, preprocess, and concatenate them
        data_list = []
        for dataset_file in dataset_files:
            print(f"\nLoading and preprocessing {dataset_file} ...")
            data = pd.read_csv(dataset_file)
            # data = preprocessing_dataset(data)
            # data = balance_dataset(data, 'emotion')
            data_list.append(data)
            data.to_csv(dataset_file, index=False)
            print(f"{dataset_file} loaded, preprocessed, balanced and saved\n")

        # Concatenate datasets
        data = pd.concat(data_list, ignore_index=True)
        print("Three datasets concatenated")

        # Print counts for each emotion
        print("\nCounts for each emotion:")
        print(data['emotion'].value_counts())

        return data

    def split_data(self, data):
        # Split data into features (text) and target (emotions)
        X = data['text']
        Y = data['emotion']
        print("Splitted data into features(text) and target(emotion)")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print("Data split into training and testing completed")

        return X_train, X_test, y_train, y_test

    def define_model_pipeline(self):
        # Define the model pipeline
        model = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(max_iter=1000, multi_class='auto')
        )
        print("\nModel pipeline defined - TfidfVectorizer, LogisticRegression")
        return model

    def train_model(self):

        dataset_files = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]

        # Load, preprocess, and balance datasets
        data = self.load_and_preprocess_datasets(dataset_files)

        # Split data into training and testing sets
        X_train, X_test, Y_train, Y_test = self.split_data(data)

        # Train the model
        self.MODEL.fit(X_train, Y_train)
        logging("info","Model trained.")

        self.save_model()

        self.evaluate_model(X_test, Y_test)

    def evaluate_model(self, X_test, Y_test):
        # Evaluate the model
        Y_pred = self.MODEL.predict(X_test)
        print(classification_report(Y_test, Y_pred))
        logging("info","Model evaluated.")

    def save_model(self):
        joblib.dump(self.MODEL, self.MODEL_NAME)
        logging("info","Model saved.")

    def load_model(self):
        model = None
        try:
            model = joblib.load(self.MODEL_NAME)
        except Exception as e:
            logging("error", str(e))
            model = self.define_model_pipeline()
        return model
            
    def predict(self, text):
        predicted_emotion = self.MODEL.predict([text])[0]
        return EmotionStates.string_to_enum(predicted_emotion).name