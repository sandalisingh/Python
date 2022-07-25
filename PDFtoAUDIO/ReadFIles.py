# python3 ReadFiles.py

import pyttsx3, time
  
def speak(text):
    # time.sleep(7)
    engine = pyttsx3.init()
    engine.setProperty("rate", 150) #178

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)

    engine.say(text)
    engine.runAndWait()

    if engine._inLoop:
        engine.endLoop()
    


# f = open("ex.txt", "w")
# f.write("Hello! How are you? Fine. Thank you.")

f = open("ex.txt", "r")
# print(f.read())
text = f.read()
speak(text)