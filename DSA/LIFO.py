#Last In First Out 
#STACK

#Sandali Singh

class Stack:
    
    #DEFAULT CONSTRUCTOR
    def __init__(self):
        self.top = -1
        self.stk = [] 

    #PUSH
    def push(self):
        ele = input("PUSH : ")
        ele = int(ele)
        self.stk.append(ele)
        self.top = self.top + 1

    #POP
    def pop(self):
        if(self.top == -1):
            print("EMPTY!")
        else:
            self.stk.pop(self.top)

    #PEEK
    def peek(self):
        if(self.top == -1):
            print("EMPTY!")
        else:
            print("PEEK : " + str(self.stk[self.top]))

    #DISPLAY THE STACK
    def display(self):
        if(self.top == -1):
            print("EMPTY!")
        else:
            print("STACK : ")
            for x in self.stk:
                print(str(x), end = " ")
            print()


#DRIVER CODE
S = Stack()

ans = "yes"
while (ans.upper() == "YES"):
    print("\nSTACK MENU : (1) Display (2) Push (3) Pop (4) Peek")
    choice = input("Enter you choice(1 to 4):\t")
    choice = int(choice)
    if choice == 1:
        S.display()
    elif choice == 2:
        S.push()
        #S.display()
    elif choice == 3:
        S.pop()
        #S.display()
    elif choice == 4:
        S.peek()
    else:
        print("\nWRONG INPUT!!!")
  #  sys.stdout.flush() 
    ans = input("Do you wish to continue?(yes/no)...\t")


