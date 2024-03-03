# WORKS
# python3 PdfToAudio.py

import pdftotext
import pyttsx3
  
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Load your PDF
with open("ex.pdf", "rb") as f:
    pdf = pdftotext.PDF(f)

# How many pages?
print("\nNumber of Pages : ",len(pdf))

# Iterate over all the pages
for page in pdf:
    print(page)
    speak(page)


# # Read some individual pages
# print(pdf[0])
# print(pdf[1])

# # Read all the text into one string
# print("\n\n".join(pdf))