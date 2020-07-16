import streamlit as st
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np
st.title(body ="Hello World")

st.write("""
 # Explore Diff
  """)


dataset =st.sidebar.selectbox("Select Datase",("Iris","Breast Cancer","Wine Dattaset"))

st.write(dataset)
classifier_name= st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))

def get_dataset(dataset_name):
    if dataset=="Iris":
        data = datasets.load_iris()
    elif dataset =="Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X= data.data
    y = data.target
    return X,y
X,y = get_dataset(dataset)
st.write("shape of datsets",X.shape)
st.write("No of classes",len(np.unique(y)))

def add_parameter_ui(clf_name):
    params=dict()
    if clf_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        params["K"]=K
    #return params
    
    elif clf_name=="SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"]=C
    #return params

    else:
        max_depth = st.sidebar.slider("max_depth",2 ,15)
        n_estimators =st.sidebar.slider("n_estimator",1,100)
        params["max_depth"] =max_depth
        params["n_estimators"]=n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    if clf_name =="KNN":
        clf= KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name=="SVM":
        clf = SVC(C=params["C"])
    else:
        clf= RandomForestClassifier(n_estimators=params["n_estimators"],
                                            max_depth=params["max_depth"],random_state=12)


    return clf

clf = get_classifier(classifier_name,params)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)


clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)

acc=accuracy_score(y_test,y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")


#plot

pca =PCA(2)
X_projected =pca.fit_transform(X)

x1=X_projected[:,0]
x2=X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha =0.8,cmap="GnBu_r")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()

#plt.show()
st.pyplot()

