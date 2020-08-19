# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:16:08 2019

@author: Nandha
"""


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder

#Importing the dataset
Diabetes="F:/Solvermind Intern/Code/Clustering/Datasets/Diabetes.csv"
dataset=pd.read_csv(Diabetes)
X=dataset.iloc[:,0:8].values
Y=dataset.iloc[:,8].values

# Find Mean and Median of the columns
a=dataset.describe()

# Histogram
dataset['Glucose'].plot(kind ="hist") 

# Correlation Matrix
corr=dataset.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(dataset.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(dataset.columns)
ax.set_yticklabels(dataset.columns)
plt.show()