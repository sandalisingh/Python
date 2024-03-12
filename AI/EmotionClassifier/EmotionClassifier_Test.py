import joblib
from EmotionClassifier import EmotionClassifier

emotion_classifier = EmotionClassifier()

while(True) :
    text = input("Text : ")
    print("Emotion :", emotion_classifier.predict(text))