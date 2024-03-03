#python3 Read_PDF.py
# 
# # importing the modules
import PyPDF2
import pyttsx3
  
# path of the PDF file
path = open("ex.pdf", "rb")
  
print("# creating a PdfFileReader object")
pdfReader = PyPDF2.PdfFileReader(path)
  
# the page with which you want to start
# this will read the page of 25th page.
print("#reading the page 1")
from_page = pdfReader.getPage(2)
  
# extracting the text from the PDF
print("extracting text from that page")
text = from_page.extractText()
print("\nEXTRACTED TEXT\n")
print(text)
print("\n\n")
  
# reading the text
print("Speak the text")
speak = pyttsx3.init()
speak.say(text)
speak.runAndWait()

print("Closing the pdf file object")
path.close()