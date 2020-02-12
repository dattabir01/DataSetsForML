import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


data = pd.read_csv('data.csv')
# SELECT -> .loc and .iloc -> location, index of location
# data.loc -> radius,perimeter; iloc -> 0,3,4
# row, columns   
data = data.iloc[:,:12]
#print(data.columns)
#print(data.info())
stats = data.describe().T
correlation = data.corr()
c2 = correlation[correlation>0.5]
#plt.figure(figsize=(11,11))
#sns.heatmap(c2, cmap='coolwarm')
#plt.show()
#plt.savefig('blah.png')
diagnosis = data.loc[:,'diagnosis']
logic_color = lambda x: 'red' if x=='M' else 'blue'

x = data.iloc[:,[2,3,4,5,6,7,8,9,10]]
y = diagnosis.map(logic_color)
#print(y[:20])
#plt.figure()
#sm = pd.plotting.scatter_matrix(x,c=y)
#plt.show()
data = data.set_index('id')
#print(data.head())
# lambda x,y:x+y
# z = (mu-x) / sigma or (mean-x)/std
stats = x.describe().T
#stats.to_csv('mystats.csv')
#x = (stats['mean']-x)/stats['std']
#x[x>3].to_csv('normalized_outliers.csv')
#now that x is normalized, we can use it for machine learning
fn = lambda v: 1 if v=='red' else 0
y = y.map(fn)
#print(y[:20])

#BEGIN MACHINE LEARNING
# SPLIT THE DATA INTO TRAINING AND TESTING
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import time

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,
                                             random_state=42)

cvs_all = []
acc_all = []
time_all = []

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

algos = {'SGD':SGDClassifier(),'SVMSimple':SVC(),'LinearSVM':
         LinearSVC(),'DTree':DecisionTreeClassifier(),'RanFor':
         RandomForestClassifier(),'KNN':KNeighborsClassifier(),
         'Sherlock':GaussianNB()}

for name,algo in algos.items():
    startime = time.time()
    clf = algo
    clf.fit(xtrain,ytrain)
    predictions = clf.predict(xtest)
    endtime = time.time()
    print(name)
    timetaken = endtime-startime
    accr = accuracy_score(predictions, ytest)
    cvscore = cross_val_score(clf, x, y, cv=5)
    print("Accuracy = " + str(accr))
    print("CVscore = " + str(cvscore))
    print("Time taken = " + str(timetaken))
    acc_all.append(accr)
    time_all.append(timetaken)
    cvs_all.append(np.mean(cvscore))












#print(len(xtrain))
#print(len(ytrain))
#print(len(xtest))
#print(len(ytest))








#bins = 12
#plt.figure(figsize=(15,15))
#for i,feature in enumerate(x.columns):
#    rows = 5
#    plt.subplot(rows,2,i+1)
#    sns.distplot(data[data['diagnosis']=='M'][feature],bins
#                 =bins, color='red',label='M')
#   sns.distplot(data[data['diagnosis']=='B'][feature],bins=
#                bins, color='blue',label='B')
#   plt.legend(loc='upper right')

#plt.tight_layout()
#plt.show()









    













