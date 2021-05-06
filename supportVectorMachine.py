import numpy as np
import pandas as pnds
import matplotlib.pyplot as plt
import seaborn as cbrn

cbrn.set(rc={'figure.figsize': [7, 7]}, font_scale=1.2)
df = pnds.read_csv('diabetes.csv')

X = df.drop(['Outcome'],axis=1)
y = df['Outcome']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
svc_lin = SVC(kernel='linear', random_state=0)
svc_lin.fit(X_train, Y_train)

y_pred = svc_lin.predict(X_train)

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score
confmtrx = confusion_matrix(Y_train, y_pred)
print(confmtrx)
print("Accuracy Score for Regression:")
print(accuracy_score(Y_train, y_pred))
print("F1 Score for Regression:")
print(f1_score(Y_train,y_pred))
