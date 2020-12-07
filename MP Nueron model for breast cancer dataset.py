Python 3.8.5 (tags/v3.8.5:580fbb0, Jul 20 2020, 15:43:08) [MSC v.1926 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> importing sklearn.datasets
SyntaxError: invalid syntax
>>> import sklearn.datasets

>>> import numpy as np
>>> b_cancer = sklearn.datasets.load_breast_cancer()
>>> b_cancer.data
array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,
        1.189e-01],
       [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,
        8.902e-02],
       [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,
        8.758e-02],
       ...,
       [1.660e+01, 2.808e+01, 1.083e+02, ..., 1.418e-01, 2.218e-01,
        7.820e-02],
       [2.060e+01, 2.933e+01, 1.401e+02, ..., 2.650e-01, 4.087e-01,
        1.240e-01],
       [7.760e+00, 2.454e+01, 4.792e+01, ..., 0.000e+00, 2.871e-01,
        7.039e-02]])
>>> b_cancer.target
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
       1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
       1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
       0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
       1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
       0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
       1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
       0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,
       1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
       1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
       1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])
>>> X.shape
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    X.shape
NameError: name 'X' is not defined
>>> X = b_cancer.data
>>> X = b_cancer.target
>>> X.shape
(569,)
>>> 
>>> Y.shape
Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    Y.shape
NameError: name 'Y' is not defined
>>> X = b_cancer.data
>>> Y = b_cancer.target
>>> import pandas as pd
>>> data = pd.DataFrame(b_cancer.data, columns = b_cancer.feature_names)
>>> data.head()
   mean radius  mean texture  ...  worst symmetry  worst fractal dimension
0        17.99         10.38  ...          0.4601                  0.11890
1        20.57         17.77  ...          0.2750                  0.08902
2        19.69         21.25  ...          0.3613                  0.08758
3        11.42         20.38  ...          0.6638                  0.17300
4        20.29         14.34  ...          0.2364                  0.07678

[5 rows x 30 columns]
>>> type(data)
<class 'pandas.core.frame.DataFrame'>
>>> data["class"].value_counts()
Traceback (most recent call last):
  File "C:\Users\surbh\AppData\Local\Programs\Python\Python38-32\lib\site-packages\pandas\core\indexes\base.py", line 2646, in get_loc
    return self._engine.get_loc(key)
  File "pandas\_libs\index.pyx", line 111, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 138, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 1619, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 1627, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'class'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#18>", line 1, in <module>
    data["class"].value_counts()
  File "C:\Users\surbh\AppData\Local\Programs\Python\Python38-32\lib\site-packages\pandas\core\frame.py", line 2800, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Users\surbh\AppData\Local\Programs\Python\Python38-32\lib\site-packages\pandas\core\indexes\base.py", line 2648, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas\_libs\index.pyx", line 111, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 138, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 1619, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 1627, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'class'
>>> data["class"] = b_cancer.target
>>> data["class"].value_counts()
1    357
0    212
Name: class, dtype: int64
>>> from sklearn.model_seletion import train_test_split
Traceback (most recent call last):
  File "<pyshell#21>", line 1, in <module>
    from sklearn.model_seletion import train_test_split
ModuleNotFoundError: No module named 'sklearn.model_seletion'
>>> from sklearn.model_selection import train_test_split
>>> X = data.drop('class', axis = 1)
>>> Y = data["class"]
>>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
>>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)
>>> import matplotlib.pyplot as plt
>>> plt.plot(X_test, '*')
[<matplotlib.lines.Line2D object at 0x1B0F6E20>, <matplotlib.lines.Line2D object at 0x1B0F6E50>, <matplotlib.lines.Line2D object at 0x1B0F6EE0>, <matplotlib.lines.Line2D object at 0x1B0F6F28>, <matplotlib.lines.Line2D object at 0x1B0F6F88>, <matplotlib.lines.Line2D object at 0x1B0F6FE8>, <matplotlib.lines.Line2D object at 0x1B106070>, <matplotlib.lines.Line2D object at 0x1B1060D0>, <matplotlib.lines.Line2D object at 0x1B106130>, <matplotlib.lines.Line2D object at 0x1B106190>, <matplotlib.lines.Line2D object at 0x1904BF70>, <matplotlib.lines.Line2D object at 0x1B1061F0>, <matplotlib.lines.Line2D object at 0x1B106280>, <matplotlib.lines.Line2D object at 0x1B1062F8>, <matplotlib.lines.Line2D object at 0x1B106358>, <matplotlib.lines.Line2D object at 0x1B1063B8>, <matplotlib.lines.Line2D object at 0x1B106418>, <matplotlib.lines.Line2D object at 0x1B106478>, <matplotlib.lines.Line2D object at 0x1B1064D8>, <matplotlib.lines.Line2D object at 0x1B106538>, <matplotlib.lines.Line2D object at 0x1B106598>, <matplotlib.lines.Line2D object at 0x1B1065F8>, <matplotlib.lines.Line2D object at 0x1B106658>, <matplotlib.lines.Line2D object at 0x1B1066B8>, <matplotlib.lines.Line2D object at 0x1B106718>, <matplotlib.lines.Line2D object at 0x1B106778>, <matplotlib.lines.Line2D object at 0x1B1067D8>, <matplotlib.lines.Line2D object at 0x1B106838>, <matplotlib.lines.Line2D object at 0x1B106898>, <matplotlib.lines.Line2D object at 0x1B1068F8>]
>>> plt.show()
>>> plt.plot(X_test.T, '*')
[<matplotlib.lines.Line2D object at 0x1AFD3640>, <matplotlib.lines.Line2D object at 0x1AFD35B0>, <matplotlib.lines.Line2D object at 0x1AFD3580>, <matplotlib.lines.Line2D object at 0x1AFD3700>, <matplotlib.lines.Line2D object at 0x1AFD3760>, <matplotlib.lines.Line2D object at 0x1AFD37C0>, <matplotlib.lines.Line2D object at 0x1AFD3820>, <matplotlib.lines.Line2D object at 0x1AFD3880>, <matplotlib.lines.Line2D object at 0x1AFD38E0>, <matplotlib.lines.Line2D object at 0x1AFD3940>, <matplotlib.lines.Line2D object at 0x1AFC26A0>, <matplotlib.lines.Line2D object at 0x1AFD39A0>, <matplotlib.lines.Line2D object at 0x1AFD3A30>, <matplotlib.lines.Line2D object at 0x1AFD3AA8>, <matplotlib.lines.Line2D object at 0x1AFD3B08>, <matplotlib.lines.Line2D object at 0x1AFD3B68>, <matplotlib.lines.Line2D object at 0x1AFD3BC8>, <matplotlib.lines.Line2D object at 0x1AFD3C28>, <matplotlib.lines.Line2D object at 0x1AFD3C88>, <matplotlib.lines.Line2D object at 0x1AFD3CE8>, <matplotlib.lines.Line2D object at 0x1AFD3D48>, <matplotlib.lines.Line2D object at 0x1AFD3DA8>, <matplotlib.lines.Line2D object at 0x1AFD3E08>, <matplotlib.lines.Line2D object at 0x1AFD3E68>, <matplotlib.lines.Line2D object at 0x1AFD3EC8>, <matplotlib.lines.Line2D object at 0x1AFD3F28>, <matplotlib.lines.Line2D object at 0x1AFD3F88>, <matplotlib.lines.Line2D object at 0x1AFD3FE8>, <matplotlib.lines.Line2D object at 0x18F21070>, <matplotlib.lines.Line2D object at 0x18F210D0>, <matplotlib.lines.Line2D object at 0x18F21130>, <matplotlib.lines.Line2D object at 0x18F21190>, <matplotlib.lines.Line2D object at 0x18F211F0>, <matplotlib.lines.Line2D object at 0x18F21250>, <matplotlib.lines.Line2D object at 0x18F212B0>, <matplotlib.lines.Line2D object at 0x18F21310>, <matplotlib.lines.Line2D object at 0x18F21370>, <matplotlib.lines.Line2D object at 0x18F213D0>, <matplotlib.lines.Line2D object at 0x18F21430>, <matplotlib.lines.Line2D object at 0x18F21490>, <matplotlib.lines.Line2D object at 0x18F214F0>, <matplotlib.lines.Line2D object at 0x18F21550>, <matplotlib.lines.Line2D object at 0x18F215B0>, <matplotlib.lines.Line2D object at 0x18F21610>, <matplotlib.lines.Line2D object at 0x18F21670>, <matplotlib.lines.Line2D object at 0x18F216D0>, <matplotlib.lines.Line2D object at 0x18F21730>, <matplotlib.lines.Line2D object at 0x18F21790>, <matplotlib.lines.Line2D object at 0x18F217F0>, <matplotlib.lines.Line2D object at 0x18F21850>, <matplotlib.lines.Line2D object at 0x18F218B0>, <matplotlib.lines.Line2D object at 0x18F21910>, <matplotlib.lines.Line2D object at 0x18F21970>, <matplotlib.lines.Line2D object at 0x18F219D0>, <matplotlib.lines.Line2D object at 0x18F21A30>, <matplotlib.lines.Line2D object at 0x18F21A90>, <matplotlib.lines.Line2D object at 0x18F21AF0>]
>>> plt.show()
>>> plt.xticks(rotation = 'vertical')
(array([0. , 0.2, 0.4, 0.6, 0.8, 1. ]), <a list of 6 Text major ticklabel objects>)
>>> plt.show()
>>> plt.show()
>>> testing_bin = X_test['mean area'].map(lambda x: 0 if x < 1000 else 1)
>>> plt.plot(testing_bin, '*')
[<matplotlib.lines.Line2D object at 0x1AC0AAF0>]
>>> plt.show()

>>> X_bin_train = X_train.apply(pd.cut, bins = 2, labels = [1,0])
>>> X_bin_test = X_test.apply(pd.cut, bins = 2, labels = [1,0])
>>> X_bin_train = X_bin_train.values #convert into numpy array
>>> X_bin_test = X_bin_test.values
>>> # MP Neuron Model
>>> 
>>> 
>>> 
>>> theta = 3
>>> from random import randint
>>> i = randint(0, X_bin_train.shape[0])
>>> if np.sum(X_bin_train[i, :])>=theta:
	print('MP Neuron says: Malignant')
    else:
	    
SyntaxError: unindent does not match any outer indentation level
>>> if np.sum(X_bin_train[i, :])>=theta:
	print('MP Neuron says: Malignant')
else:
	print('MP Neuron says: Benign')

	
MP Neuron says: Malignant
>>> if Y_train[i] == 1:   #checking if the patient was actually Malig
	print('Reality is that the patient is Malignant')
else:
	print('Reality is that the patient is Benign')

	
Reality is that the patient is Benign
>>> i
146
>>> if Y_train[i] == 1:   #checking if the patient was actually Malig
	print('Reality is that the patient is Malignant')
else:
	print('Reality is that the patient is Benign')

	
Reality is that the patient is Benign
>>> if Y_train[i] == 1:   #checking if the patient was actually Malig
	print('Reality is that the patient is Malignant')
else:
	print('Reality is that the patient is Benign')

	
Reality is that the patient is Benign
>>> y_pred_train = []
>>> correct_detection=0
>>> for x,y in zip(X_bin_train, Y_train):
	y_pred = (np.sum(x) >= theta)
	y_pred_train.append(y_pred)
	correct_detection += (y_pred == y)

	
>>> correct_detection
321
>>> # accuracy = correct_detection/total number of patients
>>> 321/512
0.626953125
>>> theta = 15
>>> y_pred_train = []correct_detection=0
SyntaxError: invalid syntax
>>> y_pred_train = []
>>> correct_detection=0
>>> for x,y in zip(X_bin_train, Y_train):
	y_pred = (np.sum(x) >= theta)
	y_pred_train.append(y_pred)
	correct_detection += (y_pred == y)

	
>>> correct_detection
324
>>> theta = 20
>>> y_pred_train = []
>>> correct_detection=0
>>> for x,y in zip(X_bin_train, Y_train):
	y_pred = (np.sum(x) >= theta)
	y_pred_train.append(y_pred)
	correct_detection += (y_pred == y)

	
>>> correct_detection
344
>>> 344/512
0.671875
>>> theta = 30
>>> y_pred_train = []correct_detection=0for x,y in zip(X_bin_train, Y_train):
	y_pred = (np.sum(x) >= theta)
	y_pred_train.append(y_pred)
	correct_detection += (y_pred == y)
	
SyntaxError: invalid syntax
>>> y_pred_train = []
>>> correct_detection=0
>>> for x,y in zip(X_bin_train, Y_train):
	y_pred = (np.sum(x) >= theta)
	y_pred_train.append(y_pred)
	correct_detection += (y_pred == y)

	
>>> correct_detection
389
>>> 389/512
0.759765625
>>> 
