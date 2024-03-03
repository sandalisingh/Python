#Circular Queue 
#Array implementation

class CircularQueue:
    #constructor
    def __init__(self, size):
        self.size = size
        self.queue = [None for i in range(size)]    #all elements ar initialised to none
        self.front = -1
        self.rear = -1  #initialising an empty queue

    def Enqueue(self):
        if((self.rear+1)%self.size == self.front):
                    print("FULL!")
        else: 
            val = input("Enter the element to be enqueued : ")
            val = int(val)

            #EMPTY  #Adding first element
            if(self.front == -1):
                self.front = 0
                self.rear = 0
                self.queue[self.rear] = val
            else:
                self.rear = (self.rear + 1) % self.size
                self.queue[self.rear] = val

    def Dequeue(self):

        #EMPTY
        if( self.front == -1):
            print("EMPTY!")

        #ONLY 1 ELEMENT 
        elif(self.front == self.rear):
            self.front = self.rear = -1     #Queue empty

        else:
            self.front = (self.front + 1) % self.size
        


    def Display(self):
        print("QUEUE : ", end = " ")
        if(self.front == -1):
            print("EMPTY!")
        elif (self.rear >= self.front):
            for i in range(self.front, self.rear+1):
                print(self.queue[i], end = " ")
        else:
            for i in range(self.front, self.size):
                print(self.queue[i], end = " ")
            for i in range(0, self.rear+1):
                print(self.queue[i], end = " ")
        print()


#DRIVER CODE
size = int(input("Enter size of the Circular Queue : "))
#size = int(size)
Q = CircularQueue(size)

ans = "yes"
while (ans.upper() == "YES"):
    print("\nCIRCULAR QUEUE MENU : (1) Display (2) Enqueue (3) Dequeue ")
    choice = input("Enter you choice(1 to 3):\t")
    choice = int(choice)
    if choice == 1:
        Q.Display()
    elif choice == 2:
        Q.Enqueue()
    elif choice == 3:
        Q.Dequeue()
    else:
        print("\nWRONG INPUT!!!")
    ans = input("Do you wish to continue?(yes/no)...\t")

