# python3 PdfToAudio2.py

# importing the modules
import PyPDF2
import pyttsx3
  
class Reading:
    # path of the PDF file
    print("\nOpening PDF file")
    path = open('ex.pdf', 'rb')
    
    # creating a PdfFileReader object
    print("creating a PdfFileReader object")
    pdfReader = PyPDF2.PdfFileReader(path)
    
    # the page with which you want to start
    # this will read the page of 25th page.
    print("Page2")
    from_page = pdfReader.getPage(2)
    
    # extracting the text from the PDF
    print("extracting the text from the PDF")
    text = from_page.extractText()

    print("\nEXTRACTED TEXT:\t"+text+"\n\n")
    
    # reading the text
    print("pyttsx3 init")
    speak = pyttsx3.init()

    if speak._inLoop:
        speak.endLoop()

    print("say")
    speak.say(text)

    print("run & wait")
    speak.runAndWait()

    print("DONE!")
    if speak._inLoop:
        speak.endLoop()