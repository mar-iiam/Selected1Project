#Processing_data
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
Categories=['99','1']#,'4','5','6','7']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir=r'C:\Users\user\Downloads\oxford-102-flower-pytorch\flower_data\flower_data\train' 
#path which contains all the categories of images
for i in Categories:
    
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(64,64,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
x=df.iloc[:,:-1] #input data 
y=df.iloc[:,-1] #output data
df.to_csv('data_64.csv')
#Model construction
from sklearn import svm
from sklearn.model_selection import GridSearchCV
#param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
param_grid={'C':[20],'gamma':[0.0001],'kernel':['rbf']}

svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)
#model training
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print('Splitted Successfully')
model.fit(x_train,y_train)
print('The Model is trained well with the given images')
# model.best_params_ contains the best parameters obtained from GridSearchCV
#model testingS
from sklearn.metrics import accuracy_score
y_pred=model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print("Confusion Matrix :",multilabel_confusion_matrix(y_test, y_pred))
r = multilabel_confusion_matrix(y_test, y_pred)
print(r)
#tn, fp, fn, tp 
# 8) Sensitivity And Specificity
sensitivity = recall_score(y_test, y_pred , average = 'macro')
specificity = recall_score(np.logical_not(y_test) , np.logical_not(y_pred) , average = 'macro')

print("Sensitivity : ",sensitivity)
print("Specificity : ",specificity)

# 9) Precision Score

print("Precision Score :",precision_score(y_test, y_pred, average='macro')) 

# 10) Recall Score
print("Recall Score :",recall_score(y_test, y_pred, average='macro'))
img_array=imread(r"C:\Users\user\Downloads\oxford-102-flower-pytorch\flower_data\flower_data\train\87\image_05475.JPG")
img_resized=resize(img_array,(64,64,3))
flat=img_resized.flatten()
y_pred=model.predict(flat.reshape(1,-1))
print(y_pred)
print(Categories[y_pred[0]])

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

