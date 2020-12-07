import sklearn.datasets
import numpy as np
import pandas as pd

b_cancer = sklearn.datasets.load_breast_cancer()
#print(b_cancer.data)
#print(b_cancer.target)
X = b_cancer.data
Y = b_cancer.target
#print(type(X))

data = pd.DataFrame(b_cancer.data, columns = b_cancer.feature_names)
#print(data.head())

data['class'] = b_cancer.target
data['class'].value_counts()

from sklearn.model_selection import train_test_split
X = data.drop('class', axis = 1)   
Y = data['class']
#print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)

import matplotlib.pyplot as plt
#plt.plot(X_test, '*')
plt.plot(X_test.T, '*')
#plt.show()
plt.xticks(rotation = 'vertical')
#plt.plot(X_test.T, '*')
#plt.show()

testing_bin = X_test['mean area'].map(lambda x:0 if x<1000 else 1)
plt.plot(testing_bin, '*')

X_bin_train = X_train.apply(pd.cut, bins = 2, labels = [1,0])
X_bin_test = X_test.apply(pd.cut, bins = 2, labels = [1,0])
X_bin_train = X_bin_train.values # convert into numpy array
X_bin_test = X_bin_test.values
cd_list=[]
# MP Neuron Model
for theta in range(1, 31):
    y_pred_train = []
    correct_detection = 0
    
    for x,y in zip(X_bin_train, Y_train):
        y_pred = (np.sum(x) >= theta)
        y_pred_train.append(y_pred)
        correct_detection += (y_pred == y)
    #print(correct_detection)
    #print("Theta = ", theta)
    
    cd_list.append(correct_detection)
#print(cd_list)
print("For training dataset")
cd = max(cd_list)
t = cd_list.index(cd)
print("theta = ",t+1)
print("Correct detection = ",cd)
print("Maximum accuracy =", (cd/512)*100,"%")
print("---------------------------------------------")

print("After applying on test dataset")

##Y_bin_train = Y_train.apply(pd.cut, bins = 2, labels = [1,0])
##Y_bin_test = Y_test.apply(pd.cut, bins = 2, labels = [1,0])
theta = 28
y_pred_test = []
correct_detection_test = 0
for a,b in zip(X_bin_test, Y_test):
    y_pred1 = (np.sum(a) >= theta)
    y_pred_test.append(y_pred1)
    correct_detection_test += (y_pred1 == b)
print("Correct detection = ",correct_detection_test)
print("Maximum accuracy of test data =", (correct_detection_test/57)*100,"%")

