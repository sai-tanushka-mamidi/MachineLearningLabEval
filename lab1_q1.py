#Write a program to count the number of vowels and consonants present in an input string.
s = input("Enter a string: ")
v = 0
c = 0
for i in s:
    if i == 'a' or i == 'e' or i == 'i' or i == 'o' or i == 'u' or \
       i == 'A' or i == 'E' or i == 'I' or i == 'O' or i == 'U':
        v = v + 1
    elif (i >= 'a' and i <= 'z') or (i >= 'A' and i <= 'Z'):
        c = c + 1

print("Vowels =", v)
print("Consonants =", c)
