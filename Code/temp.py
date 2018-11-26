# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:21:38 2018

@author: elind
"""

# Python program to count the 
# number of numbers in a given range 
import numpy as np

def count(list1, l, r): 
	
	# x for x in list1 is same as traversal in the list 
	# the if condition checks for the number of numbers in the range 
	# l to r 
	# the return is stored in a list 
	# whose length is the answer 
	return len(list(x for x in list1 if l <= x <= r)) 

# driver code 
list1 = np.array([10, 20, 30, 40, 50, 40, 40, 60, 70]) 
keys = np.array([30, 40, 50, 60])
print(count(list1, l, r))
