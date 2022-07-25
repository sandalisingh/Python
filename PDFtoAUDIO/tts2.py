#WORKS


# importing the pyttsx library
import pyttsx3
  
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
# initialisation

  
# testing

text1 = "My first code on text-to-speech"
print("SPEAK: " + text1)
speak(text1)

# text2 = "Thank you, Geeksforgeeks"
# print("\n\nSPEAK: " + text2)
# speak(text2)

text3 = input("\nENTER TEXT : ")
print("\n\nSPEAK: " + text3)
speak(text3)

text3 = input("\nENTER TEXT : ")
print("\n\nSPEAK: " + text3)
speak(text3)

text3 = input("\nENTER TEXT : ")
print("\n\nSPEAK: " + text3)
speak(text3)

text3 = input("\nENTER TEXT : ")
print("\n\nSPEAK: " + text3)
speak(text3)