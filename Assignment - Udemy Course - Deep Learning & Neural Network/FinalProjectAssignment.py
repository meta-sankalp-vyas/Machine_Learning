# -*- coding: utf-8 -*-
"""
Created on Thu May 12 21:05:18 2022

@author: Sankalp Vyas
"""

import pandas as pd
from sklearn.decomposition import PCA
 
# Headers for the CSV
columns = [
    "BI_RADS","Age","Shape","Margin","Density","Severity"
]

#Read CSV
mamographic_masses = pd.read_csv('D:\MLCourse\mammographic_masses.data.txt', names = columns, na_values = ['?'])

mamographic_masses_null = mamographic_masses[mamographic_masses.isnull().any(axis=1)]

'''Drop NA Values,
axis represent whether rows or columns needs to be removed which contains Null Values
inplace is for to make updates in the same Dataframe or not'''
mamographic_masses_NonNA = mamographic_masses.dropna(axis='rows', inplace = False)


print(mamographic_masses.head(10))
'''Difference between the Dropped and Non Dropped DataFrame Rows'''
print(mamographic_masses_NonNA.describe())
print(mamographic_masses_null.describe())


# Extract Features and Classes in different Arrays
all_features = mamographic_masses_NonNA[['Age','Shape','Density','Margin']].values
all_classes = mamographic_masses_NonNA[['Severity']].values
all_classes = all_classes.squeeze()

feature_names = ['Age','Shape','Density','Margin']

#print(all_features)

#Normalize the Data with NUMPY using sklearn preprocessing
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)
#print(all_features_scaled)

'''Differentiate the Train and Test Data. 75% for the Training and 25% for Testing.
 Using ModelSelection's train_test_split from sklearn'''
import numpy as np
np.random.seed(1234)

from sklearn.model_selection import train_test_split

(training_inputs, 
 testing_inputs, 
 training_classes, 
 testing_classes) = train_test_split(all_features_scaled, all_classes, train_size=0.75, random_state=1)

'''Decision Trees via Decision Tree Classifiers'''
#import Decision Tree Cla
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=1)
clf.fit(training_inputs, training_classes)

#Display the Decision Tree
from IPython.display import Image
import six
import sys
sys.modules['sklearn.externals.six'] = six
#from sklearn.externals.six import StringIO
from io import StringIO 
from sklearn import tree
from pydotplus import graph_from_dot_data

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names)
exported = dot_data.getvalue()
graph = graph_from_dot_data(exported)
Image(graph.create_png())

#Get the Score of the Trained Data Set
print('Trained Score : ' + (str)(clf.score(testing_inputs, testing_classes)))

#Instead of Single Test Split, lets use K-Fold Cross Validation using K = 10
#import cross val score from modelselection
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=1)
cross_valScore = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
print('Decision Tree Score : ' + (str)(cross_valScore.mean()))

#Using RandomForestClassifier to identify the output
#import RandomForestClassifier from sklearn.enseamble
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, random_state=1)
cross_valScoresRandomForest = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
print('Random Forest Score : ' + (str)(cross_valScoresRandomForest.mean()))

'''Using Support Virtual Machine(SVM)'''
#From sklearn we will import svm, will provide regularization parameter as 1 and 
#will do SVC function to create an estimator
from sklearn import svm
clf = svm.SVC(kernel='linear', C=1.0)
cross_valScoreSVCLinear = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
print('SVC Linear Score : ' + (str)(cross_valScoreSVCLinear.mean()))

clf = svm.SVC(kernel='sigmoid', C=1.0)
cross_valScoreSVCSigmoid = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
print('SVC Sigmoid Score : ' + (str)(cross_valScoreSVCSigmoid.mean()))

clf = svm.SVC(kernel='poly', C=1.0)
cross_valScoreSVCPoly = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
print('SVC Poly Score : ' + (str)(cross_valScoreSVCPoly.mean()))

clf = svm.SVC(kernel='rbf', C=1.0)
cross_valScoreSVCRBF = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
print('SVC RBF Score : ' + (str)(cross_valScoreSVCRBF.mean()))


'''K Nearest Neighbours'''
#Import Nearest Neighbous classifiers and predict
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors=10)
cross_valScoreKNN = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
print('KNN Score :' + (str)(cross_valScoreKNN.mean()))

'''for i in range(1, 50):
    k=i
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    cross_valScoreKNN = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
    print('KNN Score :', i, ' : ', (cross_valScoreKNN.mean()))'''
    
'''Naive Bayes'''
from sklearn.naive_bayes import MultinomialNB
scaler = preprocessing.MinMaxScaler()
all_features_scaledMinMax = scaler.fit_transform(all_features)
clf = MultinomialNB()

cross_valScoreNaiveBayes = cross_val_score(clf, all_features_scaledMinMax, all_classes, cv=10)
print('Naive Bayes : ', cross_valScoreNaiveBayes.mean()) 


'''Logistic Regression'''
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
cross_valScoreLogisticRegression = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
print('Logistic Regression : ', cross_valScoreLogisticRegression.mean())

'''Binary Keras Neural Network'''
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def create_model():
    model = Sequential()
    model.add(Dense(6, activation = 'relu', input_dim = 4, kernel_initializer = 'normal'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer = 'normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(build_fn=create_model, epochs = 100, verbose = 0)
cv_scoreNN = cross_val_score(estimator, all_features_scaled, all_classes, cv=10)
print('Neural Network Score : ', cv_scoreNN.mean())