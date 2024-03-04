#Question 1
def count_vowels(word):
    vowels ="AEIOUaeiou"
    num = 0
    for i in word:
        if i in vowels:
            num+=1
    return num
#Question 2
def enlarge(lst):
    animals=['tiger', 'elephant', 'monkey', 'zebra', 'panther']
    lst = animals
    for element in range(len(lst)):
        if lst[element].lower() == True:
            lst[element] = lst[element].upper()
    return lst


#Question 3
def even(n,n= 20):
    for i in range(n):
        if i%2 == 0:
            print(i,"even")
        else:
            print(i,"odd")
#Question 4
def sum_of_integers(a, b):
    return a+b
