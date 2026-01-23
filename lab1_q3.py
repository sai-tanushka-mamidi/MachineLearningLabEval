# Write a program to find the number of common elements between two lists. The lists 
#contain integers.
list1 = input("Enter numbers for the first list").split()
list1 = [int(x) for x in list1]  
list2 = input("Enter numbers for the second list").split()
list2 = [int(x) for x in list2]  
common_elements = []
for number in list1:
    if number in list2:          
        common_elements.append(number)
print("Common elements are:", common_elements)
print("Number of common elements:", len(common_elements))
