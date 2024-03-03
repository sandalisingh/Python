File = open("ReadingWritingFiles.txt", "a")

for i in range(10):
    File.write("\n\t<<< " + str(i))

File.close()