import string
count=0

def check_length(pwd):#Checks the length of the password
    global count
    if len(pwd) < 8:
        print("--> Password Size must be >= 8")
    else:
        count += 1

def check_upper(pwd):#Checks if there is at least one upper case letter
    global count
    found=False
    for char in pwd:
        if char.isupper():
            found = True
            break
    if found:
        count += 1
    else:
        print("--> Password must contain at least one uppercase letter")


def check_lower(pwd):# Check if password has at least one lowercase letter
    global count
    found=False
    for char in pwd:
        if char.islower():
            found=True
            break
    if found:
        count+=1
    else:
        print("--> Password must contain at least one lowercase letter")



def check_digit(pwd):# Check for at least one  digit
    global count
    found=False
    for char in pwd:
        if char.isdigit():
            found = True
            break
    if found:
        count += 1
    else:
        print("--> Password must contain at least one digit")



def check_special(pwd):# Check if password has at least one special character
    global count
    special_characters = string.punctuation
    found = False
    for char in pwd:
        if char in special_characters:
            found = True
            break
    if found:
        count += 1
    else:
        print("--> Password must contain at least one special character")
    
def check_password_strength(pwd):
    check_length(pwd)
    check_upper(pwd)
    check_lower(pwd)
    check_digit(pwd)
    check_special(pwd)


def main():
    global count
    count=0
    password=input("Enter The password you want to check")
    check_password_strength(password)
    if count <= 2:
        print("🔴 Weak Password")
    elif count == 3 or count == 4:
        print("🟡 Medium Password")
    else:
        print("🟢 Strong Password")

if __name__=="__main__":
    main()

    
        
        
        
        
    
    



