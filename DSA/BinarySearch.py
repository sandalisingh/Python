
#recursive function BINARY SEARCH
def BinarySearch(arr, start, end, ele):
    
    if start <= end:
        mid = (start + end)//2
        if arr[mid] == ele: #found the element
            return mid+1    #returning its position
        elif arr[mid] > ele:    #consider left subarray
            return BinarySearch(arr, start, mid-1, ele)
        else:   #arr[mid] < ele #consider right subarray
            return BinarySearch(arr, mid+1, end, ele)

    return -1   #elemenet not found
    
    

a = []  #creates list like arraylist in java 
size = input("Enter size of array:  ")  #inputs size in string
size = int(size)    #converts string to int

i = 0
print("Enter array elements(in ascending order):\n")
while i < size:
    j = input("A[" + str(i) + "] = ")   #inputs array elements in string
    j = int(j)  #converts string to int
    a.append(j) #appends the input element at the end of the list
    i = i + 1   #increements i
print(a)    #prints list 

ele = input("\nEnter the element to be searched:\t")    #inputs the element to be searched in string
ele = int(ele)  #converts string to int
position = BinarySearch(a, 0, size, ele)    #calls the BinarySearch function and srores the return value in int variable 

if position != -1:  
    print("Found at "+str(position)+"th postion in the array")  #displays the position of the elemenet
else:   #BinarySearch fxn returns -1 if the element to be searched is not found
    print("\nNOT FOUND!!!")



