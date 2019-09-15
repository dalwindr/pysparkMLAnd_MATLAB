import math as m
from math import *
from turtledemo.chaos import plot
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sb
# #%matplotlib inline
#
# #plot(arange(5))
#
# df = pd.read_csv("/Users/keeratjohar2305/Downloads/Dataset/AVtest_LoanPrediction.csv") #Reading the dataset in a dataframe using Pandas
# df.shape
# print(df.head(10))
# print(df.describe())
# print(df['Property_Area'].value_counts())
# df['ApplicantIncome'].hist(bins=50)

plt3 = plt
# 3. multiple figure

data = np.arange(100, 201)
plt3.plot(data)

data2 = np.arange(200, 301)
plt3.figure()
plt3.plot(data2)

plt3.show()
exit(1)


#
pltT = plt
# evenly sampled time at 200ms intervals
#np.arange(0, 100, 5)  => (start, end, difference b/w two values)
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
pltT.plot(t , t, 'r--', t, t**2, 'bs', t, t**3, 'y^')
pltT.show()


exit(1)
# example 6 bar charts
N = 7

plt9 = plt

data = np.random.randint(low=0, high=100, size=N)
x = np.arange(len(data))
colors = np.random.rand(N * 3).reshape(N, -1)
labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
print(" x : ", x)
print(" data : ", data)
print(" colors : ", colors)
print(" labels : ", labels)

plt9.title("Weekday Data")
plt9.bar(x, data)#, alpha=0.8, color=colors, tick_label=labels)
plt9.show()
plt9.legend()


exit(1)

# example 6 bar histogram
plt11 = plt
print(np.random.randint(0, 10000, 13)) # Generate 13 numbers randoms under 10 Thousand line

arr1 = np.random.randint(0, 10, 10)
arr2 = np.random.randint(0, 20, 10)
arr3 = np.random.randint(0, 30, 10)
print(arr1)
print(arr2)
print(arr3)

data = [arr1,arr2,arr3]
print(data)
labels = ['3K', '4K', '5K']
bins = [0, 5, 10, 15, 20]

plt11.hist(data, bins=bins, label=labels)
plt11.show()
exit(1)
#PART 3
plt10 = plt
print(np.random.randint(0, 10000, 13)) # Generate 13 numbers randoms under 10 Thousand line


data = [np.random.randint(0, n, n) for n in [3000, 4000, 5000]]
print(data)
labels = ['3K', '4K', '5K']
bins = [0, 100, 500, 1000, 2000, 3000, 4000, 5000]

plt10.hist(data, bins=bins, label=labels)
plt10.show()


plt9 = plt

exit(1)
plt8 = plt
# example 6 pie charts

labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ListOf7RandomNumber = np.random.rand(7)
data = ListOf7RandomNumber * 100

plt8.pie(data, labels=labels, autopct='%1.1f%%')
plt8.axis('equal')
plt8.legend()
plt8.show()


# Example 5. scatter plot
# part 7 Scatter subplot
plt7 = plt
randomNumList1 = np.random.rand(20)
randomNumList2 = np.random.rand(20)
randomNumList3 = np.random.rand(20)
randomNumList4 = np.random.rand(20)
randomNumList5 = np.random.rand(20)
randomNumList6 = np.random.rand(20)

plt7.subplot(311)
plt7.scatter(randomNumList1 * 100,
             randomNumList2 * 100,
             c='r', s=100, alpha=0.5)
plt7.subplot(312)
plt7.scatter(randomNumList3 * 100,
             randomNumList4 * 100,
             c='g', s=200, alpha=0.5)
plt7.subplot(313)
plt7.scatter(randomNumList5 * 100,
             randomNumList6 * 100,
             c='b', s=300, alpha=0.5)

plt7.show()


# part 1
plt6 = plt
randomNumList1 = np.random.rand(20)
randomNumList2 = np.random.rand(20)
randomNumList3 = np.random.rand(20)
randomNumList4 = np.random.rand(20)
randomNumList5 = np.random.rand(20)
randomNumList6 = np.random.rand(20)

plt6.scatter(randomNumList1 * 100,
             randomNumList2 * 100,
             c='r', s=100, alpha=0.5)

plt6.scatter(randomNumList3 * 100,
             randomNumList4 * 100,
             c='g', s=200, alpha=0.5)

plt6.scatter(randomNumList5 * 100,
             randomNumList6 * 100,
             c='b', s=300, alpha=0.5)

plt6.show()


# part 1
plt5 = plt
randomNumList = np.random.rand(20)
plt5.subplot(3, 1, 1)
plt5.scatter(randomNumList * 100,
             randomNumList * 100,
             c='r', s=100, alpha=0.5)

plt5.scatter(randomNumList * 100,
             randomNumList * 100,
             c='g', s=200, alpha=0.5)

plt5.scatter(randomNumList * 100,
             randomNumList * 100,
             c='b', s=300, alpha=0.5)

plt5.show()

exit(1)
plt4 = plt
# Example 4  Linear graph

plt4.plot([1, 2, 3], [3, 6, 9], '-r')
plt4.plot([1, 2, 3], [2, 4, 9], ':g')

plt4.show()






plt1 = plt
plt2 = plt


#  2. Multiple subplots
data = np.arange(100, 201)
plt1.subplot(3, 1, 1)
plt1.plot(data)
data2 = np.arange(110, 301)
plt1.subplot(3, 1, 2)
plt1.plot(data2)

data3 = np.arange(200, 301)
plt1.subplot(3, 1, 3)
plt1.plot(data3)

plt1.show()

# 1. co graph

data = np.arange(100, 201)
plt2.plot(data)

data2 = np.arange(110, 301)
plt2.plot(data2)

data3 = np.arange(200, 301)
plt2.plot(data3)

plt2.show()

