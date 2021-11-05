import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree
from tabulate import tabulate


df = pd.read_csv('https://raw.githubusercontent.com/Bijay555/Decision-Tree-Classifier/master/iris.csv')
print(df.head)

df.describe()

X = df[['Sepal.Length','Sepal.Width', 'Petal.Length', 'Petal.Width']].values
y = df['Species'].values

(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=1)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc.score(X_test, y_test)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
y_pred

acc=accuracy_score(y_test,y_pred)*100
print ("Accuracy is :")
print(acc)
results = np.array(confusion_matrix(y_test, y_pred) )
print ('Confusion Matrix :')
print(results)
cls_report = classification_report(y_test,y_pred)
print(cls_report) 


tr=pickle.dump(dtc, open("model.pkl", "wb"))



