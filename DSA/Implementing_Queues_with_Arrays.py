#IMPLEMENTING QUEUES WITH ARRAYS
#Sandali Singh 

class Queue:
    #default constructor
    def __init__(self, n):
        self.size = n
        self.front = -1
        self.rear = -1
        self.Que = []   #creating a list

    def Enqueue(self):
        ele = input("\nEnter the element you wish to enqueue:\t")
        ele = int(ele)
        print("ENQUEUE...\n")
        if (self.rear == self.size - 1) :  #rear is pointing to last element in array, array is full
            print("FULL!!!")
        else:
            self.Que.append(ele)    #adding the new element at the end of the list
            self.rear += 1  #rear points to the added new element in the list
            if(self.front == -1):   #array was initially empty
                self.front = 0  #first element is added, front points to the 0th element

    def Dequeue(self):
        print("\nDEQUEUE...\n")
        if(self.rear == -1): 
            print("EMPTY!!!")
        else:
            print(str(self.Que.pop(0)) + "\thas been removed from the Queue!!!")
            if(self.front == self.rear == 0):
                self.front = self.rear = -1
            else:
                self.rear -= 1
            

    def Display(self):
        print("\nQUEUE:\t")
        if(self.rear == -1):
            print("EMPTY!!!")
        else:
            for i in self.Que:
                print(str(i), end = " <- ")
            print()

    def Front(self):
        if (self.front == -1):
            print("\nEMPTY!!!")
        else:
            print("\nElement at FRONT:\t"+str(self.Que[self.front]))



#DRIVER CODE

size = input("Enter the CAPACITY of the QUEUE:\t")
size = int(size)
Q = Queue(size)

ans = "yes"
while (ans.upper() == "YES"):
    print("\nQUEUE: MENU\t1.Display\t2.Enqueue\t3.Dequeue\t4.Front")
    choice = input("Enter you choice(1 to 4):\t")
    choice = int(choice)
    if choice == 1:
        Q.Display()
    elif choice == 2:
        Q.Enqueue()
        #Q.Display()
    elif choice == 3:
        Q.Dequeue()
        #Q.Display()
    elif choice == 4:
        Q.Front()
    else:
        print("\nWRONG INPUT!!!")
  #  sys.stdout.flush() 
    ans = input("Do you wish to continue?(yes/no)...\t")


