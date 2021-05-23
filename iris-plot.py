#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:39:50 2021

@author: oyamatoshiki
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import csv

lst = []
l = 1
for i in np.linspace(-2.0, 3.0, 30):
    for j in np.linspace(-2.0, 2.0, 30):
        for k in np.linspace(-2.0, 2.0, 30):
            lst.append([i, j, k, l])
            

with open("/Users/sample/mesh-iris.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(lst)

data = pd.read_csv("/Users/sample/mesh-iris.csv",  header = None)
data = data.dropna()
data.columns = ['X', 'Y', 'Z', 'label']

XYZ = data.iloc[0:, [0,1,2]].values

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.loc[df['target'] == 0, 'target'] = "setosa"
df.loc[df['target'] == 1, 'target'] = "versicolor"
df.loc[df['target'] == 2, 'target'] = "virginica"

class_mapping = {label:idx for idx, label in enumerate(np.unique(df['target']))}

df['target'] = df['target'].map(class_mapping)

y0 = df.iloc[0:, 4].values


X0 = df.iloc[0:, [0,2,3]].values

CL_lis = ['red','blue','green']

MK_lis = ['o','v','^']

LB_lis = ['setosa','versicor','virginica']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
#標準化
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std  = sc.transform(X_test)

from sklearn.svm import SVC

svm = SVC(kernel = 'linear', C=10, random_state=0)

svm.fit(X_train_std,y_train)

y_pred = svm.predict(X_test_std)

y_pred2 = svm.predict(XYZ)

X0_std = sc.transform(X0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("sepal length")
ax.set_ylabel("petal length")
ax.set_zlabel("petal width")


ax.scatter(XYZ[y_pred2 == 0, 0], XYZ[y_pred2 == 0, 1], XYZ[y_pred2 == 0, 2], 
          color = 'red', marker = '+', label = "T1(scatter)", linestyle = 'None', s = 40, alpha = 0.4)

ax.scatter(XYZ[y_pred2 == 1, 0], XYZ[y_pred2 == 1, 1], XYZ[y_pred2 == 1, 2],
           color = 'blue', marker = '+', label = "T2(scatter)", linestyle = 'None', s = 40, alpha = 0.4)

ax.scatter(XYZ[y_pred2 == 2, 0], XYZ[y_pred2 == 2, 1], XYZ[y_pred2 == 2, 2],
           color = 'green', marker = '+', label = "T3(scatter)", linestyle = 'None', s = 40, alpha = 0.4)

angle = 30+90

#45度ずつ

ax.view_init(30, angle) 




