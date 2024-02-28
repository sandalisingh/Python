import joblib

# Load the saved model
model = joblib.load('emotion_classifier.joblib')

def emotion_classifier(text):
    predicted_emotion = model.predict([text])[0]
    return predicted_emotion

while(True) :
    text = input("Text : ")
    print("Emotion :", emotion_classifier(text))