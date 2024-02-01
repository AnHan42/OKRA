import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import os
import glob as gb
import cv2
from sklearn import decomposition, preprocessing, svm
import Kernel_lib

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x    
        

trainpath = "/home/sc.uni-leipzig.de/jf994pyjo/OKRA/data/dataset2-master/dataset2-master/images/TRAIN"
testpath = "/home/sc.uni-leipzig.de/jf994pyjo/OKRA/data/dataset2-master/dataset2-master/images/TEST"
predpath = "/home/sc.uni-leipzig.de/jf994pyjo/OKRA/data/dataset2-master/dataset2-master/images/TEST_SIMPLE"

for folder in os.listdir(trainpath): 
    folder_path = os.path.join(trainpath, folder)
    files = gb.glob(pathname=os.path.join(folder_path, '*.jpeg'))
    print(f'For training data, found {len(files)} in folder {folder}')

code = {'EOSINOPHIL':0 ,'LYMPHOCYTE':1,'MONOCYTE':2,'NEUTROPHIL':3}


#reading the images
s = 100
X_train = []
y_train = []
for folder in  os.listdir(trainpath) : 
    folder_path = os.path.join(trainpath, folder)
    files = gb.glob(pathname=os.path.join(folder_path, '*.jpeg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_train.append(list(image_array))
        y_train.append(code[folder])


X_test = []
y_test = []
for folder in  os.listdir(testpath) : 
    folder_path = os.path.join(trainpath, folder)
    files = gb.glob(pathname=os.path.join(folder_path, '*.jpeg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_test.append(list(image_array))
        y_test.append(code[folder])

X_train = np.array(X_train)
X_test = np.array(X_test)
# X_pred_array = np.array(X_pred)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Assuming your data is stored in X_train
X_train_shape = X_train.shape

# Reshape the data to 2D array where each row represents a single picture
X_train = X_train.reshape(X_train_shape[0], -1)

X_text_shape = X_test.shape

# Reshape the data to 2D array where each row represents a single picture
X_test = X_test.reshape(X_text_shape[0], -1)

partitions, y = Kernel_lib.partition_dataset(X_train, y_train, 2)

k = 10
n_features = partitions[0].shape[1] 

# OKRA kernel
gamma = Kernel_lib.get_gamma(n_features, 1, seed=42)
A_prime = partitions[0] @ gamma.T
B_prime = partitions[1] @ gamma.T

full = Kernel_lib.compute_gram_matrix(A_prime, B_prime)

clf = svm.SVC(kernel = 'precomputed', probability=True)
clf.fit(Kernel_lib.polynomial_kernel(full, degree=3), y)

clf3 = svm.SVC(kernel = 'poly', probability = True)
clf3.fit(X_train, y_train)

print('Blood cells: Performance on Naive Kernel: ')
Kernel_lib.metrics_micro(clf3, X_train, y_train, cv=5)

print('Blood cells: Performance of OKRA with Custom (Polynomial) Gram Kernel: ')
Kernel_lib.metrics_micro(clf, full, y_train, cv=5)