# # importing required modules 
# import PyPDF2 
    
# # creating a pdf file object 
# pdfFileObj = open('ex.pdf', 'rb') 

  
# # creating a pdf reader object 
# pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
    
# # printing number of pages in pdf file 
# print("No of pages : ",pdfReader.numPages) 
    
# # creating a page object 
# pageObj = pdfReader.getPage(0) 
    

# print("\nText in 1st page\n") 
# # extracting text from page 

# print("Extracting text from page 1 of the pdf")
# print(pageObj.extractText()) 
    
# # closing the pdf file object 
# print("Closing the pdf file close")
# pdfFileObj.close() 



import slate3k as slate

with open("ex.pdf",'rb') as f:
    extracted_text = slate.PDF(f)
print(extracted_text)