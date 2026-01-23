#Generate a list of 100 random numbers between 100 and 150. Find the mean, median and 
#mode for these numbers.
import numpy as np
numbers=np.random.randint(100, 151, 100)
mean=np.mean(numbers)
median=np.median(numbers)
mode= 3 * median- 2 * mean
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
